# Import libraries
import joblib
from sklearn.metrics import accuracy_score

def make_prediction(text, pipeline_path):
    # Loading pipeline
    pipeline = joblib.load(pipeline_path)
    vectorizer = pipeline['vectorizer']
    classifier = pipeline['classifier']

    # Adjusting inputs format
    if isinstance(text, str):
        text = [text]

    preprocessed_text = vectorizer.transform(text)
    classifications = classifier.predict(preprocessed_text)
    return classifications

if __name__ == '__main__':
    reviews = [
        # 50 positive
        "A beautifully acted, emotionally resonant story that stayed with me long after the credits.",
        "Clever screenplay, sharp dialogue, and a lead performance that carries the entire film.",
        "Visually stunning with inventive camera work — one of the best-looking films I've seen this year.",
        "Perfect balance of humor and heart; the pacing never drags and the jokes land.",
        "An ambitious indie that succeeds thanks to strong writing and memorable characters.",
        "The soundtrack elevates every scene, and the director's vision is crystal clear.",
        "I loved the worldbuilding and how the film rewards attention to small details.",
        "Surprising and thought-provoking — it challenged my expectations in the best way.",
        "Terrific supporting cast; every role felt fully realized and meaningful.",
        "Tight editing and a gripping story made this a can't-miss thriller.",
        "A warm, funny family movie that also handles real emotional stakes.",
        "Outstanding special effects paired with a surprisingly human story.",
        "Great chemistry between the leads; their relationship felt authentic and lived-in.",
        "An affirmative, uplifting film with real depth and strong performances.",
        "Smart, witty, and emotionally satisfying — a rare triple threat.",
        "A masterclass in suspense; the director knows how to build tension.",
        "Charming and original — I haven't seen anything quite like this.",
        "Sharp social commentary woven into an engaging plot. Highly recommended.",
        "An old-school heist movie updated with fresh twists and excellent pacing.",
        "Beautiful score and gorgeous production design — a feast for the senses.",
        "The writing is subtle and layered; it rewards repeat viewings.",
        "A tender coming-of-age story with believable dialogue and strong heart.",
        "Perfect casting choices — each actor adds depth to the story.",
        "Inventive visuals and a haunting score make this one unforgettable.",
        "Funny, smart, and surprisingly profound — will watch again.",
        "A bold, original sci-fi that focuses on character rather than just spectacle.",
        "Impeccable timing in both drama and comedy moments; everything clicks.",
        "A nuanced portrayal of complicated relationships — honest and affecting.",
        "Powerful performances and a script that refuses to take easy routes.",
        "The film handles heavy themes with grace and never feels preachy.",
        "Crisp direction and a clever script make this a real crowd-pleaser.",
        "Warmly acted, beautifully shot, and emotionally satisfying from start to finish.",
        "A tight, clever thriller with a twist I didn't see coming.",
        "Playful, fast-paced, and full of heart — great entertainment.",
        "A deeply moving drama that avoids melodrama; performances are exquisite.",
        "Fantastic worldbuilding and clear stakes — felt invested the whole time.",
        "The humor is smart and well-timed, plus the cast has great chemistry.",
        "A tender, genuine romance that avoids clichés and feels fresh.",
        "Bold filmmaking and a killer central performance make this a must-see.",
        "This documentary is informative, well-structured, and emotionally resonant.",
        "An exhilarating ride from start to finish — I was on the edge of my seat.",
        "Powerful, intimate, and beautifully acted — one of the year's best.",
        "Clever script, great performances, and a finale that pays off.",
        "A rare artistry in modern filmmaking — subtle and deeply affecting.",
        "The pacing is excellent, and the film never wastes a single moment.",
        "Hilarious and heartfelt; it balances comedy and emotion perfectly.",
        "Strong thematic core and a director who clearly cares about the material.",
        "An inventive thriller that uses its premise to explore real human questions.",
        "I appreciated the restraint — it trusts the audience and still delivers.",
        "A satisfying, well-crafted film that I would happily recommend to friends.",

        # 50 negative
        "Predictable plot that telegraphs every twist — felt like a waste of potential.",
        "The pacing is painfully slow and the characters never feel interesting.",
        "Poorly written dialogue that made it hard to care about anyone on screen.",
        "Visual effects were inconsistent and often distracting from the story.",
        "An uneven mess — strong idea at the center but executed clumsily.",
        "The film tries to do too much and ends up shallow in every direction.",
        "Bad casting choices; the leads have no chemistry and the performances fall flat.",
        "A contrived script full of clichés and emotional manipulation.",
        "The soundtrack was overpowering and the editing was choppy.",
        "This movie is all style and no substance — pretty to look at but empty.",
        "Overlong and self-indulgent; it never justifies its runtime.",
        "Confusing narrative structure that left me more frustrated than intrigued.",
        "Attempts at humor miss the mark and feel forced throughout.",
        "Predictable rom-com beats with slapdash jokes and thin characters.",
        "A disappointing sequel that recycles old ideas without innovation.",
        "The action scenes are incoherent and poorly choreographed.",
        "Fails to commit to a tone — is it drama, comedy, or thriller? None work.",
        "A hollow biopic that glosses over the most interesting parts of its subject.",
        "The film leans on gimmicks instead of building real emotional stakes.",
        "Awful pacing and a finale that undercuts everything that came before.",
        "Poor production design and uninspired cinematography make it forgettable.",
        "The characters' decisions are nonsensical, purely serving contrived plot points.",
        "Did not land emotionally — manipulative rather than honest.",
        "Soap-opera melodrama dressed up as serious filmmaking.",
        "Flat, monotonous delivery from the entire cast.",
        "Heavy-handed message with no subtlety or nuance.",
        "The premise had promise, but the screenplay is shallow and unfocused.",
        "Too many subplots that clutter the main story and confuse the audience.",
        "Weak villain and stakes that never feel real or interesting.",
        "Cheap-looking effects and amateurish direction throughout.",
        "Novelty premise can't save the thin characters and forgettable lines.",
        "The narrative is full of plot holes and lazy exposition.",
        "An overlong documentary filled with repetition and no real insights.",
        "I couldn't connect to any character — the film keeps them at arm's length.",
        "A messy tone and clumsy shifts between comedy and tragedy.",
        "Dialogue is filled with clichés and predictable one-liners.",
        "The ending is unsatisfying and feels tacked on.",
        "Poorly paced thriller that squanders tension with bad choices.",
        "Attempts at artistry feel pretentious and hollow.",
        "The emotional beats are earned poorly and feel unearned.",
        "Stilted performances and direction that keeps actors from breathing.",
        "A bland story that feels like a checklist of genre tropes.",
        "It wants to be profound but only achieves shallow platitudes.",
        "The remake adds nothing new and removes what made the original good.",
        "Inconsistent visual style and sloppy continuity mistakes.",
        "The film's humor is mean-spirited rather than clever.",
        "A forgettable thriller with no compelling characters or stakes.",
        "Technical problems — sound mixing issues made dialogue hard to follow.",
        "The emotional core is missing; it never gives you a reason to care.",
        "An undercooked idea stretched into a feature with nothing to justify it.",
    ]
    labels = [1]*50 + [0]*50

    preds_rp = make_prediction(reviews,'models/tfidf_svm_review_polarity.joblib')
    preds_pd = make_prediction(reviews, 'models/tfidf_svm_rt-polaritydata.joblib')
    preds_imdb = make_prediction(reviews, 'models/tfidf_svm_aclImdb.joblib')

    print("Prediction accuracy for 100 artificially generated samples:")
    print(accuracy_score(labels, preds_rp), "for review polarity")
    print(accuracy_score(labels, preds_pd), "for polarity data")
    print(accuracy_score(labels, preds_imdb), "for IMDb")