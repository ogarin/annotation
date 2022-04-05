import transformers


# This is a test for debugging BART generation


pipeline = transformers.pipeline("summarization", "Salesforce/bart-large-xsum-samsum")

dialogue = """Wiley Customer Service : Thank you for contacting Wiley Customer Service . My name is Erez . How may I assist you today ? 
Ori : May I please be emailed a copy of your company w-9 ? Thanks . 
Ori : I need to create an order .   You are not a vendor with us .   We need to add you . 
Wiley Customer Service : Thank you for that information ! 
Ori : ogarin@salesforce.com
Wiley Customer Service : I will be more than happy to assist you . 
Wiley Customer Service : Please , hold . 
Wiley Customer Service : Thank you for holding ! This incident is being escalated to the next support group for further investigation .   A specialist from this group will be contacting you via email in response to this incident within 24 to 48 business .
"""

print("=" * 20 + "  Without sampling  " + "=" * 20)
res = pipeline(dialogue, num_return_sequences=6)
print("\n".join(
    r['summary_text'] for r in res))

print("\n" + "=" * 20 + "  With sampling  " + "=" * 20)
res = pipeline(dialogue, num_return_sequences=6, do_sample=True)
print("\n".join(
    r['summary_text'] for r in res))


print("\n" + "=" * 20 + "  With sampling top_p=0.8 " + "=" * 20)
res = pipeline(dialogue, num_return_sequences=6, do_sample=True, top_p=0.8)
print("\n".join(
    r['summary_text'] for r in res))



# res = pipeline(dialogue, num_beams=15, num_return_sequences=15, output_scores=True,
#                return_dict_in_generate=True)
# print(res)
# print("\n".join(
#     f"{score:.4f} {sum}" for sum, score in zip(res.sequences, res.sequence_scores)
# ))
#
