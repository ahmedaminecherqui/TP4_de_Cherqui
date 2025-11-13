package ma.emsi.cherqui.td4_de_cherqui;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Paths;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;

public class RagNaif {
    public static void main(String[] args) {

        // 1. Récupère la clé secrète depuis une variable d'environnement
        String apiKey = System.getenv("GEMINI_API_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException(
                    "La clé API Gemini n'est pas définie dans la variable d'environnement GEMINI_API_KEY"
            );
        }

        // 2. Crée le modèle de chat Gemini
        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.7)
                .timeout(Duration.ofSeconds(60))
                .build();

        // 3. Crée la mémoire (garde jusqu'à 10 messages)
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatMemory(chatMemory)
                .chatModel(chatModel)
                .build();

        // 1.Récupération du fichier PDF à utiliser comme source
        Path cheminFichier = Paths.get("src/main/resources/support_rag.pdf"); // exemple

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


    }
}
