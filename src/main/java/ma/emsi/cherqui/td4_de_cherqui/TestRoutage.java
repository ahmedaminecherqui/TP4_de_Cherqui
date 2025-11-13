package ma.emsi.cherqui.td4_de_cherqui;

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
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class TestRoutage {

    // ---------------------------------------------------
    // Méthode utilitaire pour éviter duplication
    // ---------------------------------------------------
    private static void ingest(Path path, EmbeddingStore<TextSegment> store, EmbeddingModel embModel) {

        var document = FileSystemDocumentLoader.loadDocument(
                path,
                new ApacheTikaDocumentParser()
        );

        var segments = DocumentSplitters.recursive(500, 100).split(document);

        for (var segment : segments) {
            Embedding embedding = embModel.embed(segment.text()).content();
            store.add(embedding, segment);
        }
    }

    public static void main(String[] args) {
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

        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(apiKey)
                .modelName("text-embedding-004")
                .build();

        // 2.Crée la mémoire (garde jusqu'à 10 messages)
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        // ---------------------------------------
        // 2) Phase 1 – INGESTION des 2 documents
        // ---------------------------------------

        EmbeddingStore<TextSegment> storeCours = new InMemoryEmbeddingStore<>();
        EmbeddingStore<TextSegment> storeAutre = new InMemoryEmbeddingStore<>();

        ingest(Path.of("src/main/resources/langchain_langchain4j.pdf"), storeCours, embeddingModel);
        ingest(Path.of("src/main/resources/Génie Logiciel et Qualité du Logiciel.pdf"), storeAutre, embeddingModel);

        // ---------------------------------------
        // 3) PHASE 2 – RETRIEVERS
        // ---------------------------------------

        ContentRetriever retrieverCours = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeCours)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .build();

        ContentRetriever retrieverAutre = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeAutre)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .build();

        // ---------------------------------------
        // 4) QUERY ROUTER (Gemini decides the route)
        // ---------------------------------------

        Map<ContentRetriever, String> descriptions = new HashMap<>();
        descriptions.put(
                retrieverCours,
                "Documents sur Langchain and Langchain4j."
        );
        descriptions.put(
                retrieverAutre,
                "Documents sur Genie et Qualité Logiciel"
        );

        QueryRouter router = new LanguageModelQueryRouter(
                chatModel,
                descriptions
        );

        // ---------------------------------------
        // 5) RETRIEVAL AUGMENTOR
        // ---------------------------------------

        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        // ---------------------------------------
        // 6) RAG Assistant
        // ---------------------------------------

        Assistant assistant = AiServices.builder(Assistant.class)
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
