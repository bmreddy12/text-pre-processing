from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

nltk.download("book")
from nltk.book import *

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker_tab")
nltk.download("words")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

text = """
    A lot of the data that you could be analyzing is unstructured data and contains human-readable text.
    Before you can analyze that data programmatically, you first need to preprocess it.
    In this tutorial, you’ll take your first look at the kinds of text preprocessing tasks you can do with NLTK so that you’ll be ready to apply them in future projects.
    You’ll also see how to do some basic text analysis and create visualizations.
    By tokenizing, you can conveniently split up text by word or by sentence.
    This will allow you to work with smaller pieces of text that are still relatively coherent and meaningful even outside of the context of the rest of the text.
    It’s your first step in turning unstructured data into structured data, which is easier to analyze.
    """
lotr_quote = "It's a dangerous business, Frodo, going out your door."
words_in_lotr_quote = word_tokenize(lotr_quote)
lotr_pos_tags = nltk.pos_tag(words_in_lotr_quote)


def sentence_tokenize():
    print(f"Tokenize in Sentence:\n{sent_tokenize(text)}\n")


def words_tokenize():
    print(f"Tokenize in Words:\n{word_tokenize(text)}\n")


def filter_stop_words():
    worf_quote = "Sir, I protest. I am not a merry man!"
    words_in_quote = word_tokenize(worf_quote)
    filtered_list = []
    for word in words_in_quote:
        if word.casefold() not in stop_words:
            filtered_list.append(word)
    print(f"Filter Stop words:\n{filtered_list}\n")


def stemming():
    string_for_stemming = """
    The crew of the USS Discovery discovered many discoveries.
    Discovering is what explorers do."""
    words = word_tokenize(string_for_stemming)
    stemmed_words = [stemmer.stem(word) for word in words]
    print(f"Stemmed Words:\n{stemmed_words}\n")


def tag_pos():
    sagan_quote = """
    If you wish to make an apple pie from scratch,
    you must first invent the universe."""
    jabberwocky_excerpt = """
    'Twas brillig, and the slithy toves did gyre and gimble in the wabe:
    all mimsy were the borogoves, and the mome raths outgrabe."""

    words_in_sagan_quote = word_tokenize(sagan_quote)
    words_in_excerpt = word_tokenize(jabberwocky_excerpt)
    print(f"POS:\n{nltk.pos_tag(words_in_sagan_quote)}\n")
    print(f"POS with gibberish words: \n{nltk.pos_tag(words_in_excerpt)}\n")


def lemmatizing():
    string_for_lemmatizing = "The friends of DeSoto love scarves."
    words = word_tokenize(string_for_lemmatizing)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    print(f"Lemmatized words:\n{lemmatized_words}\n")
    # To lemmatize noun to adjective of word 'worst'
    print(lemmatizer.lemmatize("worst", pos="a"))


def chunking():
    print(f"words with POS tag:\n{lotr_pos_tags}")
    # chunk grammar
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    chunk_parser = nltk.RegexpParser(grammar)
    tree = chunk_parser.parse(lotr_pos_tags)
    tree.draw()


def chinking():
    grammar = """
    Chunk: {<.*>+}
    }<JJ>{"""
    chunk_parser = nltk.RegexpParser(grammar)
    tree = chunk_parser.parse(lotr_pos_tags)
    tree.draw()


def named_entity_recognition():
    tree = nltk.ne_chunk(lotr_pos_tags)
    tree.draw()
    # To know what the named entities are but not what kind of named entity they are
    tree = nltk.ne_chunk(lotr_pos_tags, binary=True)
    tree.draw()


def concordance():
    # concordance: extract each time a word is used, along with its immediate context
    text8.concordance("man")


def dispersion_plot():
    # Use a dispersion plot to see how much a particular word appears and where it appears
    text2.dispersion_plot(["Allenham", "Whitwell", "Cleveland", "Combe"])
    plt.show()


def frequency_distribution():
    # frequency distribution: Can check which words show up most frequently in the text
    frequency_distribution = FreqDist(text8)
    print(f"Frequency Distribution: \n{frequency_distribution}\n")

    # to see the 20 most common words in the corpus
    print(f"20 Most common Frequency Distribution: \n{frequency_distribution.most_common(20)}\n")

    # list of all the words in text8 that aren’t stop words
    meaningful_words = [word for word in text8 if word.casefold() not in stop_words]
    frequency_distribution = FreqDist(meaningful_words)
    print(f"20 Most common Frequency Distribution that aren’t stop words: \n{frequency_distribution.most_common(20)}\n")


def collocations():
    # Collocation is a sequence of words that shows up often
    text8.collocations()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in text8]
    new_text = nltk.Text(lemmatized_words)
    new_text.collocations()


if __name__ == '__main__':
    print(
        "\nChoose from the list\n1.sentence tokenize\n2.words tokenize\n3.filter stop words\n4.stemming\n5.tag POS"
        "\n6.Lemmatizing\n7.Chunking\n8.Chinking\n9.NER\n10.Concordance\n11.Dispersion Plot\n12.Frequency Distribution"
        "\n13.Collocations\n14.EXIT\n"
    )
    while True:
        print("Enter option(eg:1 for sentence tokenize)\n")
        option = int(input("Option: "))
        if 1 <= option <= 14:
            match option:
                case 1:
                    sentence_tokenize()
                case 2:
                    words_tokenize()
                case 3:
                    filter_stop_words()
                case 4:
                    stemming()
                case 5:
                    tag_pos()
                case 6:
                    lemmatizing()
                case 7:
                    chunking()
                case 8:
                    chinking()
                case 9:
                    named_entity_recognition()
                case 10:
                    concordance()
                case 11:
                    dispersion_plot()
                case 12:
                    frequency_distribution()
                case 13:
                    collocations()
                case 14:
                    exit()
        else:
            print("Invalid Option")
