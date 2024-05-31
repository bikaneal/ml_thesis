nltk.download('stopwords')
nltk.download('punkt')

model = Word2Vec.load('data\word2vec-classification-50')

with open('data\classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)
    
def preprocess(text, stop_words, punctuation_marks, morph):
    tokens = word_tokenize(text.lower())
    preprocessed_text = []
    for token in tokens:
        if token not in punctuation_marks:
            lemma = morph.parse(token)[0].normal_form
            if lemma not in stop_words:
                preprocessed_text.append(lemma)
    return preprocessed_text

morph = pymorphy3.MorphAnalyzer()
stop_words = stopwords.words('russian')
punctuation_marks = ['!', ',', '(', ')', ':', '-', '?', '.', '..', '...', '«', '»']

def document_vector(word2vec_model, doc):
    # Отфильтровываем слова, которых нет в модели
    doc = [word for word in doc if word in word2vec_model.wv]
    if len(doc) == 0:
        return np.zeros(word2vec_model.vector_size, dtype=np.float32)

    # Считаем среднее всех векторов слов документа
    return np.mean(word2vec_model.wv[doc], axis=0)

def get_vector(model, tokenized_text):
    return document_vector(model, tokenized_text)

def process_text(text):
    speakers = {}
    lines = text.split('\n')  # каждая новая строка считается репликой

    for line in lines:
        if not line.strip():
            continue  # Пропускаем пустые строки

        try:
            speaker, utterance = line.split(': ', 1)
        except ValueError:  # если строка не содержит спикера
            continue

        speaker = speaker.strip()
        utterance = utterance.strip()

        if speaker not in speakers:
            speakers[speaker] = []

        # Предобработка реплики перед подачей в модель
        preprocessed_utterance = preprocess(utterance, stop_words, punctuation_marks, morph)

        # Получение вектора для отдельно взятой реплики
        new_vector = get_vector(model, preprocessed_utterance)
        new_vector = np.array([new_vector])

        # Получение предсказания модели
        prediction = classifier.predict(new_vector)

        if prediction:
            # Принадлежность реплики данному спикеру уже гарантирована, добавляем ее в список
            speakers[speaker].append(utterance)

    return speakers

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    
    # Вызываем функцию обработки текста, передаем ей текст и другие необходимые параметры
    result = process_text(text)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)