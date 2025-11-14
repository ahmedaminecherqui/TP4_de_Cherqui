package ma.emsi.cherqui.td4_de_cherqui;

import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.embedding.EmbeddingModel;


import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;

import dev.langchain4j.data.embedding.Embedding;

import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;


import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;



import dev.langchain4j.service.AiServices;

import java.nio.file.Path;
import java.time.Duration;
import java.util.*;

public class TestPasRag {
    // -------------------------------
    // Méthode utilitaire ingestion
    // -------------------------------
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

        // -------------------------------
        // 1) Créer le modèle Gemini
        // -------------------------------
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

        // -------------------------------
        // 2) Ingestion du document
        // -------------------------------
        EmbeddingStore<TextSegment> storeCours = new InMemoryEmbeddingStore<>();
        ingest(Path.of("src/main/resources/Génie Logiciel et Qualité du Logiciel.pdf"), storeCours, embeddingModel);

        ContentRetriever retrieverCours = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeCours)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .build();

        // -------------------------------
        // 3) QueryRouter personnalisé : utiliser RAG ou pas
        // -------------------------------
        QueryRouter ragRouter = new QueryRouter() {
            @Override
            public Collection<ContentRetriever> route(Query query) {
                // Prompt simple : pas de PromptTemplate, juste String concat
                String prompt = "Est-ce que la requête \"" + query + "\" porte sur les concepts du genie logiciel et la qualité lgiciel ? "
                        + "Réponds seulement par 'oui', 'non' ou 'peut-être'.";

                // Poser la question au LM directement
                ChatRequest chatRequest = ChatRequest.builder()
                        .messages(UserMessage.from(prompt))
                        .build();

                String lmResponse = chatModel.chat(chatRequest)
                        .aiMessage()
                        .text()
                        .toLowerCase();


                if (lmResponse.contains("oui") || lmResponse.contains("peut-être")) {
                    return List.of(retrieverCours);
                } else {
                    return Collections.emptyList();
                }
            }
        };



        // -------------------------------
        // 4) RetrievalAugmentor
        // -------------------------------
        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(ragRouter)
                .build();

        // -------------------------------
        // 5) Assistant RAG
        // -------------------------------
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(chatModel)
                .retrievalAugmentor(augmentor)
                .build();

        // -------------------------------
        // 6) Test
        // -------------------------------
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


