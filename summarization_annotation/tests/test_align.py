import align
import spacy


# This is a test for debugging alignment in summvis


nlp = spacy.load("en_core_web_lg")
dialogue = """Wiley Customer Service : Thank you for contacting Wiley Customer Service . My name is Erez . How may I assist you today ? 
Ori : May I please be emailed a copy of your company w-9 ? Thanks . 
Ori : I need to create an order .   You are not a vendor with us .   We need to add you . 
Wiley Customer Service : Thank you for that information ! 
Ori : ogarin@salesforce.com
Wiley Customer Service : I will be more than happy to assist you . 
Wiley Customer Service : Please , hold . 
Wiley Customer Service : Thank you for holding ! This incident is being escalated to the next support group for further investigation .   A specialist from this group will be contacting you via email in response to this incident within 24 to 48 business .
"""
summary = """
Issue : The customer would like a copy of Wiley 's company w-9 form in order for the school district to create a PO .

Resolution : Escalated
"""
pred = """
Issue : The customer would like a form.
Resolution : the case was escalated
"""

from datasets import load_metric
metric = load_metric("rouge")
def preprocess(text):
#     return re.sub("\n+", "\n", text, re.MULTILINE)
    doc = nlp(text)
    return '\n'.join(
        ' '.join(token.text for token in sentence)
            for sentence in doc.sents
    )


import bert_score
res = bert_score.score([preprocess(pred)], [preprocess(summary)],
                       lang="en", return_hash=True, rescale_with_baseline=True,
                       idf=False, verbose=True)
print(res)

res = align.BertscoreAligner(0.2, 10).align(nlp(dialogue), [nlp(summary)])
print(res)
