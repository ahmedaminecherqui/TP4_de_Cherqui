package ma.emsi.cherqui.td4_de_cherqui;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.nio.file.Paths;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestRagavecRecuperation {
    private static void configureLogger() {
        // Configure le logger sous-jacent (java.util.logging)
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE); // Ajuster niveau
        // Ajouter un handler pour la console pour faire afficher les logs
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }
    public static void main(String[] args) {

        configureLogger();

        // === Phase 1 : Enregistrement des embeddings ===

        //Récupère la clé secrète depuis une variable d'environnement
        String apiKey = System.getenv("GEMINI_API_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException(
                    "La clé API Gemini n'est pas définie dans la variable d'environnement GEMINI_API_KEY"
            );
        }

        //Crée le modèle de chat Gemini
        ChatLanguageModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.7)
                .timeout(Duration.ofSeconds(60))
                .logRequestsAndResponses(true)
                .build();



        // 1.Récupération du fichier PDF à utiliser comme source
        Path cheminFichier = Paths.get("src/main/resources/langchain_langchain4j.pdf"); // exemple

        //2.Création du parser Apache Tika (aucun argument requis)
        ApacheTikaDocumentParser parser = new ApacheTikaDocumentParser();

        // 3. Charger le document avec FileSystemDocumentLoader
        Document document = FileSystemDocumentLoader.loadDocument(cheminFichier, parser);

        // 4. Découper le document
        var splitter = DocumentSplitters.recursive(500, 100);
        List<TextSegment> segments = splitter.split(document);

        // 5. Créer un modèle d'embedding Gemini
        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(apiKey)
                .modelName("text-embedding-004")
                .build();

        // 6. Créer les embeddings pour chaque segment
        Response<List<Embedding>> response = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = response.content();

        // 7. Ajouter les embeddings dans un magasin en mémoire
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);

        // === Phase 2 : Création du ContentRetriever ===

        // 1.creation du content retriver
        ContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(2)             // on ne garde que les 2 segments les plus pertinents
                .minScore(0.5)       // uniquement si le score >= 0.5
                .build();

        // 2.Crée la mémoire (garde jusqu'à 10 messages)
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);



        // === Phase 3 : Ajout du WebSearch (Tavily) ===

        // 1. On récupère la clé Tavily
        String tavilyKey = System.getenv("TAVILY_API_KEY");
        if (tavilyKey == null || tavilyKey.isBlank()) {
            throw new IllegalStateException(
                    "La clé API Tavily n'est pas définie dans la variable TAVILY_API_KEY"
            );
        }

        // 2. Création du moteur de recherche Web Tavily
        var tavilyEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyKey)
                .build();

        // 3. Création d'un ContentRetriever basé sur le Web
        ContentRetriever webRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(tavilyEngine)
                .maxResults(3) // nombre de résultats Web
                .build();

        // === Phase 4 : QueryRouter combinant PDF + Web ===

        var router = new DefaultQueryRouter(
                retriever,     // celui du PDF
                webRetriever   // celui du Web
        );

        // === Phase 5 : Création du RetrievalAugmentor ===

        var augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        // Création de l’assistant avec le RetrieavalAugmentor
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatMemory(chatMemory)
                .chatLanguageModel(chatModel)
                .retrievalAugmentor(augmentor)
                .build();


        // Ask a question that must be answered from infos.txt
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question : ");
                String question = scanner.nextLine();
                if (question.isBlank()) {
                    continue;
                }
                System.out.println("==================================================");
                if ("fin".equalsIgnoreCase(question)) {
                    break;
                }
                String reponse = assistant.chat(question);
                System.out.println("Assistant : " + reponse);
                System.out.println("==================================================");
            }
        }

    }
}

