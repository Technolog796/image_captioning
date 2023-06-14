import string


def truncate_sentences(sentences):
    truncated_sentences = []
    exclude = set(string.punctuation)
    for sentence in sentences:
        truncated_sentence = sentence
        index = truncated_sentence.find("<pad>")
        if index != -1:
            truncated_sentence = truncated_sentence[:index]
        truncated_sentence = ''.join(ch for ch in truncated_sentence if ch not in exclude)
        index = truncated_sentence.find("бродить")
        if index != -1:
            truncated_sentence = truncated_sentence[index + 8:]
        truncated_sentences.append(truncated_sentence)
    return truncated_sentences
