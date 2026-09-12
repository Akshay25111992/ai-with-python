# add sentence-transformers in requirement.txt

from sentence_transformers import SentenceTransformer

def find_top_similar_sentences(input_sentence, reference_sentences, top_k=2):
    # Load the SBERT model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Generate embeddings for all reference sentences
    reference_embeddings = model.encode(
        reference_sentences,
        normalize_embeddings=True
    )

    # Generate embedding for the input sentence
    input_embedding = model.encode(
        input_sentence,
        normalize_embeddings=True
    )

    # Calculate cosine similarity
    similarities = model.similarity(
        input_embedding,
        reference_embeddings
    )[0]

    # Create (score, sentence) pairs
    results = []

    for sentence, score in zip(reference_sentences, similarities):
        results.append(
            (float(score), sentence)
        )

    # Sort by highest similarity
    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return results[:top_k]


def main():

    reference_sentences = [
        "How can I reset my password?",
        "What is the process for changing my password?",
        "How do I update my email address?",
        "How can I cancel my order?",
        "What is the refund policy?"
    ]

    # Input parameter
    input_sentence = "I forgot my password and want to change it"

    top_results = find_top_similar_sentences(
        input_sentence,
        reference_sentences,
        top_k=2
    )

    print("Input:")
    print(input_sentence)

    print("\nTop 2 Similar Sentences:")
    print("-" * 60)

    for rank, (score, sentence) in enumerate(top_results, start=1):
        print(f"{rank}. Score: {score:.4f}")
        print(f"   {sentence}")
        print()


if __name__ == "__main__":
    main()