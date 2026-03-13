import re
import nltk
import spacy
import unicodedata
from nltk import TweetTokenizer
from spacy.lang.es import Spanish
from spacy.lang.en import English
from nltk.util import ngrams


# Mapa de emojis comunes a texto con carga semántica de sentimiento
EMOJI_SENTIMENT_MAP = {
    '😊': 'emoji_feliz', '😀': 'emoji_feliz', '😁': 'emoji_feliz',
    '😂': 'emoji_risa', '🤣': 'emoji_risa',
    '😍': 'emoji_amor', '❤️': 'emoji_amor', '💕': 'emoji_amor', '♥': 'emoji_amor',
    '😢': 'emoji_triste', '😭': 'emoji_triste', '💔': 'emoji_triste',
    '😡': 'emoji_enojo', '😠': 'emoji_enojo', '🤬': 'emoji_enojo',
    '👍': 'emoji_positivo', '✅': 'emoji_positivo', '🙌': 'emoji_positivo',
    '👎': 'emoji_negativo', '❌': 'emoji_negativo',
    '😴': 'emoji_aburrido', '🙄': 'emoji_hastio',
    '😱': 'emoji_sorpresa', '😮': 'emoji_sorpresa',
    '✨': 'emoji_positivo', '🎉': 'emoji_celebracion',
}


class TextProcessing(object):
    name = 'Text Processing'
    lang = 'es'

    def __init__(self, lang: str = 'es'):
        self.lang = lang
        self.nlp = TextProcessing.load_spacy(lang=lang)

    @staticmethod
    def load_spacy(lang: str):
        result = None
        try:
            if lang == 'es':
                result = spacy.load('es_core_news_sm')
            else:
                result = spacy.load('en_core_web_sm')
            print('Language: {0}\n{1}: {2}'.format(TextProcessing.name, lang, result.pipe_names))
        except Exception as e:
            print('Error load_spacy: {0}'.format(e))
        return result

    def analysis_pipe(self, text: str):
        doc = None
        try:
            doc = self.nlp(text=text)
        except Exception as e:
            print('Error analysis_pipe: {0}'.format(e))
        return doc

    @staticmethod
    def proper_encoding(text: str):
        result = ''
        try:
            text = unicodedata.normalize('NFD', text)
            text = text.encode('ascii', 'ignore')
            result = text.decode("utf-8")
        except Exception as e:
            print('Error proper_encoding: {0}'.format(e))
        return result

    @staticmethod
    def stopwords(text: str, lang: str = 'es'):
        """
        Elimina stopwords del texto.
        CORRECCIÓN: el original comparaba la clase con el string 'es' (siempre False),
        por lo que nunca seleccionaba correctamente el idioma.
        """
        result = ''
        try:
            # BUG ORIGINAL: `if TextProcessing == 'es'` compara la clase, no el idioma
            # CORRECCIÓN: usar el parámetro lang recibido
            nlp = Spanish() if lang == 'es' else English()
            doc = nlp(text)
            token_list = [token.text for token in doc]
            sentence = []
            for word in token_list:
                lexeme = nlp.vocab[word]
                if not lexeme.is_stop:
                    sentence.append(word)
            result = ' '.join(sentence)
        except Exception as e:
            print('Error stopwords: {0}'.format(e))
        return result

    @staticmethod
    def remove_patterns(text: str):
        result = ''
        try:
            text = re.sub(r'\©|\×|\⇔|\»|\«|\~|\#|\$|\€|\Â|\®|\¬', '', text)
            text = re.sub(r'\,|\;|\:|\!|\¡|\'|\'|\"|\"|\"|\'|\`', '', text)
            text = re.sub(r'\}|\{|\[|\]|\(|\)|\<|\>|\?|\¿|\°|\|', '', text)
            text = re.sub(r'\/|\-|\+|\*|\=|\^|\%|\&|\$', '', text)
            text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)
            # Eliminar '_' solo cuando NO está entre letras (preserva hashtag_x, emoji_x)
            text = re.sub(r'(?<![a-z])_|_(?![a-z])', '', text)
            result = text.lower()
        except Exception as e:
            print('Error remove_patterns: {0}'.format(e))
        return result

    @staticmethod
    def replace_emojis_semantic(text: str):
        """
        NUEVO: Reemplaza emojis conocidos por tokens semánticos de sentimiento
        en lugar del genérico [EMOJI]. Preserva señal útil para clasificación.
        Los emojis no mapeados se eliminan.
        """
        result = text
        try:
            for emoji_char, token in EMOJI_SENTIMENT_MAP.items():
                result = result.replace(emoji_char, f' {token} ')
            # Eliminar emojis restantes no mapeados
            result = re.sub(r"[\U0001f000-\U000e007f]", '', result)
        except Exception as e:
            print('Error replace_emojis_semantic: {0}'.format(e))
        return result

    @staticmethod
    def extract_hashtag_text(text: str):
        """
        NUEVO: Extrae el texto de los hashtags preservando su contenido semántico.
        Ejemplo: #FelizLunes → felizlunes (conserva la palabra, elimina el #)
        En lugar de reemplazarlo por [HASTAG] (token vacío de significado).
        """
        result = text
        try:
            result = re.sub(r'#([A-Za-z0-9_]{1,40})', r'hashtag_\1', result)
        except Exception as e:
            print('Error extract_hashtag_text: {0}'.format(e))
        return result

    @staticmethod
    def transformer(text: str, stopwords: bool = False, lang: str = 'es',
                    preserve_hashtags: bool = True, preserve_emojis: bool = True):
        """
        Pipeline principal de transformación de texto.

        Cambios respecto al original:
        - preserve_hashtags=True: extrae el texto del hashtag en lugar de borrarlo
        - preserve_emojis=True: mapea emojis a tokens semánticos (emoji_feliz, etc.)
        - lang: parámetro propagado correctamente a stopwords()
        - Typo corregido: [HASTAG] → [HASHTAG] cuando no se preserva

        Parámetros
        ----------
        text             : str  — tweet crudo
        stopwords        : bool — si True, elimina stopwords
        lang             : str  — idioma ('es' o 'en')
        preserve_hashtags: bool — si True, conserva texto del hashtag
        preserve_emojis  : bool — si True, mapea emojis a tokens semánticos
        """
        result = ''
        try:
            text_out = text

            # 1. Emojis — ANTES de proper_encoding (que destruye chars no-ASCII)
            if preserve_emojis:
                text_out = TextProcessing.replace_emojis_semantic(text_out)
            else:
                text_out = re.sub(r"[\U0001f000-\U000e007f]", '[EMOJI]', text_out)

            # 2. Hashtags — ANTES de proper_encoding y ANTES de remove_patterns
            #    (remove_patterns elimina '#', lo que haría que extract_hashtag_text
            #     ya no encuentre el patrón '#palabra')
            if preserve_hashtags:
                text_out = TextProcessing.extract_hashtag_text(text_out)
            else:
                text_out = re.sub(r"#([A-Za-z0-9_]{1,40})", '[HASHTAG]', text_out)

            # 3. Normalización de encoding — ya sin emojis ni hashtags en crudo
            text_out = TextProcessing.proper_encoding(text_out)
            text_out = text_out.lower()

            # 4. URLs
            text_out = re.sub(
                r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+'
                r'|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)'
                r'|[^\s`!()\[\]{};:\'".,<>?«»""]))',
                '[URL]', text_out)

            # 5. Menciones — se anonimiza, poco valor semántico en TASS
            text_out = re.sub(r"@([A-Za-z0-9_]{1,40})", '[MENTION]', text_out)

            # 6. Limpieza de patrones especiales
            #    El '#' en remove_patterns ya no afecta hashtags (procesados en paso 2)
            text_out = TextProcessing.remove_patterns(text_out)

            # 7. Stopwords (con lang propagado correctamente)
            if stopwords:
                text_out = TextProcessing.stopwords(text_out, lang=lang)

            # 8. Normalizar espacios
            text_out = re.sub(r'\s+', ' ', text_out).strip().rstrip()
            result = text_out if text_out != ' ' else None

        except Exception as e:
            print('Error transformer: {0}'.format(e))
        return result

    @staticmethod
    def tokenizer(text: str):
        val = []
        try:
            text_tokenizer = TweetTokenizer()
            val = text_tokenizer.tokenize(text)
        except Exception as e:
            print('Error tokenizer: {0}'.format(e))
        return val

    @staticmethod
    def make_ngrams(text: str, num: int):
        result = ''
        try:
            n_grams = ngrams(nltk.word_tokenize(text), num)
            result = [' '.join(grams) for grams in n_grams]
        except Exception as e:
            print('Error make_ngrams: {0}'.format(e))
        return result

    def tagger(self, text: str):
        """
        CORRECCIÓN: era @staticmethod pero llamaba a self.analysis_pipe().
        Ahora es método de instancia correctamente.
        """
        result = None
        try:
            list_tagger = []
            doc = self.analysis_pipe(text=text)   # <- corregido: self en lugar de clase
            for token in doc:
                item = {
                    'text': token.text,
                    'lemma': token.lemma_,
                    'pos': token.pos_,
                    'tag': token.tag_,
                    'dep': token.dep_,
                    'shape': token.shape_,
                    'is_alpha': token.is_alpha,
                    'is_stop': token.is_stop,
                    'is_digit': token.is_digit,
                    'is_punct': token.is_punct
                }
                list_tagger.append(item)
            result = list_tagger
        except Exception as e:
            print('Error tagger: {0}'.format(e))
        return result