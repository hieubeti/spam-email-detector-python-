# test.py
import time
import os
import spam_detector as sd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def checkpoint(msg):
    print(f">>> {msg} ({time.strftime('%H:%M:%S')})")

def mainFunction():
    print(">>> FILE RUN:", __file__)

    checkpoint("CHECKPOINT A: loading test filenames...")
    test_fileNames = sd.get_testFileNames()
    print("    ✔ Loaded:", len(test_fileNames), "test emails")

    checkpoint("CHECKPOINT B: counting emails...")
    nb_of_allEmails = sd.number_of_allEmails()
    nb_of_spamEmails = sd.number_of_spamEmails()
    nb_of_hamEmails = sd.number_of_hamEmails()
    print("    ✔ Total:", nb_of_allEmails, "| Spam:", nb_of_spamEmails, "| Ham:", nb_of_hamEmails)

    checkpoint("CHECKPOINT C: generating training words...")
    all_trainWords, spam_trainWords, ham_trainWords = sd.trainWord_generator()
    print("    ✔ C DONE")

    checkpoint("CHECKPOINT D: unique words...")
    all_uniqueWords = sd.unique_words(all_trainWords)
    print("    ✔ Unique words:", len(all_uniqueWords))

    checkpoint("CHECKPOINT E: bag of words...")
    spam_bagOfWords, ham_bagOfWords = sd.bagOfWords_genarator(all_uniqueWords, spam_trainWords, ham_trainWords)
    print("    ✔ E DONE")

    checkpoint("CHECKPOINT F: smoothing...")
    smoothed_spamBOW, smoothed_hamBOW = sd.smoothed_bagOfWords(all_uniqueWords, spam_bagOfWords, ham_bagOfWords, 0.5)
    print("    ✔ F DONE")

    checkpoint("CHECKPOINT G: probabilities...")
    spam_prob = sd.spam_probability(nb_of_allEmails, nb_of_spamEmails)
    ham_prob = sd.ham_probability(nb_of_allEmails, nb_of_hamEmails)
    print("    ✔ G DONE")

    checkpoint("CHECKPOINT H: conditional probabilities...")
    print("      → Computing spam_condProbability...")
    spam_condProb = sd.spam_condProbability(all_uniqueWords, spam_bagOfWords, smoothed_spamBOW, 0.5)
    print("      → Computing ham_condProbability...")
    ham_condProb = sd.ham_condProbability(all_uniqueWords, ham_bagOfWords, smoothed_hamBOW, 0.5)
    print("    ✔ H DONE")

    checkpoint("CHECKPOINT I: actual labels...")
    actual_labels = sd.get_actualLabels()
    print("    ✔ I DONE")

    checkpoint("CHECKPOINT J: scoring...")
    ham_scores, spam_scores, predicted_labels, decision_labels = sd.score_calculator(
        all_uniqueWords, spam_prob, ham_prob, spam_condProb, ham_condProb, 0.5)
    print("    ✔ J DONE")

    fileNumbers = len(test_fileNames)

    checkpoint("CHECKPOINT K: creating result.txt ...")
    result_output = sd.result_output_generator(
        fileNumbers, test_fileNames, predicted_labels, ham_scores, spam_scores, actual_labels, decision_labels)
    result_path = os.path.join(BASE_DIR, "result.txt")
    with open(result_path, "w", encoding="utf8") as f:
        f.write(result_output)
    print("    ✔ result.txt CREATED at:", result_path)

    checkpoint("CHECKPOINT L: evaluation...")
    spam_precision = sd.get_spamPrecision(fileNumbers, actual_labels, predicted_labels)
    spam_recall = sd.get_spamRecall(fileNumbers, actual_labels, predicted_labels)
    spam_accuracy = sd.get_spamAccuracy(fileNumbers, actual_labels, predicted_labels)
    spam_fmeasure = sd.get_spamFmeasure(spam_precision, spam_recall)

    ham_precision = sd.get_hamPrecision(fileNumbers, actual_labels, predicted_labels)
    ham_recall = sd.get_hamRecall(fileNumbers, actual_labels, predicted_labels)
    ham_accuracy = sd.get_hamAccuracy(fileNumbers, actual_labels, predicted_labels)
    ham_fmeasure = sd.get_hamFmeasure(ham_precision, ham_recall)

    evaluation_result_output = sd.evaluation_result(
        spam_accuracy, spam_precision, spam_recall, spam_fmeasure,
        ham_accuracy, ham_precision, ham_recall, ham_fmeasure)

    spam_tp, spam_tn, spam_fp, spam_fn = sd.spamConfusionParams(fileNumbers, actual_labels, predicted_labels)
    spam_confusionMatrix_output = sd.spam_confusionMatrix(spam_tp, spam_tn, spam_fp, spam_fn)

    ham_tp, ham_tn, ham_fp, ham_fn = sd.hamConfusionParams(fileNumbers, actual_labels, predicted_labels)
    ham_confusionMatrix_output = sd.ham_confusionMatrix(ham_tp, ham_tn, ham_fp, ham_fn)

    evaluation_output = sd.evaluation_output_generator(evaluation_result_output, spam_confusionMatrix_output, ham_confusionMatrix_output)

    checkpoint("CHECKPOINT M: creating evaluation.txt ...")
    eval_path = os.path.join(BASE_DIR, "evaluation.txt")
    with open(eval_path, "w", encoding="utf8") as f:
        f.write(evaluation_output)
    print("    ✔ evaluation.txt CREATED at:", eval_path)

    print("\nSummary:")
    print("  - model.txt (if you ran train.py) ->", os.path.join(BASE_DIR, "model.txt"))
    print("  - result.txt ->", result_path)
    print("  - evaluation.txt ->", eval_path)
    print("  - If result/eval missing: check that 'train' and 'test' folders contain files and names include 'spam' or 'ham' for labels.")

if __name__ == "__main__":
    mainFunction()
