from transformers import T5Tokenizer, T5ForConditionalGeneration
from tool import get_topics
import spacy
from sentence_transformers import SentenceTransformer


def get_question(answer, context, max_length=64):
  input_text = "answer: %s  context: %s </s>" % (answer, context)
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=max_length)

  return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_questions_from_cluster(data):
    quiz_output = {}
    for clusters in data:
        for topic, sentences in clusters.items():
            quiz_output[topic] = []
            for sentence in sentences:
                doc = nlp(sentence)
                # Pick top noun chunks or named entities as potential "answers"
                candidates = [chunk.text for chunk in doc.noun_chunks]
                if not candidates:
                    candidates = [ent.text for ent in doc.ents]
                for answer in candidates[:1]:  # Limit to 1 question per sentence to avoid overload
                    try:
                        question = get_question(answer, sentence)
                        quiz_output[topic].append({
                            "question": question,
                            "answer": answer,
                            "context": sentence
                        })
                    except Exception as e:
                        print(f"Failed to generate question for: {answer} | {sentence}")
                        continue
    return quiz_output


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    tokenizer = T5Tokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
    model = T5ForConditionalGeneration.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    notesData = get_topics('sample_notes.pdf', embedder)
    questions = generate_questions_from_cluster(notesData)
    print(questions)