import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


def generate_doc(
        i, 
        topicnames, 
        docnames, 
        words, 
        df_doc_topic, 
        df_topic_word, 
        min_words, 
        max_words
    ):

    docname = docnames[i]
    doc = []
    num_words = np.random.randint(
        min_words, max_words
    )
    top_pro = df_doc_topic.loc[docname, :]
    topic_list = [np.random.choice(topicnames, p=top_pro) for _ in range(num_words)]
    for topic in topic_list:
        word_pro = df_topic_word.loc[topic, :]
        word = np.random.choice(words, p=word_pro)
        doc.append(word)
    doc = " ".join(doc)
    return doc


def generate_docs_by_lda(
        num_docs, 
        num_topics, 
        num_covs, 
        doc_topic_prior,
        lambda_,
        sigma,
        min_words, 
        max_words, 
        voc_size, 
        num_jobs, 
        seed
    ):

    np.random.seed(seed)

    topicnames = ["Topic" + str(i) for i in range(num_topics)]
    docnames = ["Doc" + str(i) for i in range(num_docs)]
    words = ["word_" + str(i) for i in range(voc_size)]

    beta = [0.1 for _ in range(voc_size)]

    if num_covs > 0:
        M_prevalence_covariates = np.random.randint(2,size=(num_docs, num_covs))

    doc_topic_pro_list = []
    for i in range(num_docs):
        if doc_topic_prior == "dirichlet":
            if num_covs > 0:
                alpha = np.exp(np.dot(M_prevalence_covariates[i, :], lambda_))
            else:
                alpha = np.array([0.1 for _ in range(num_topics)])
            doc_topic_pro_list.append(np.random.dirichlet(alpha))
        elif doc_topic_prior == "logistic_normal":
            if num_covs > 0:
                mean = np.dot(M_prevalence_covariates[i, :], lambda_)
            else:
                mean = np.array([0 for _ in range(num_topics)])
                sigma = np.eye(num_topics)
            sample = np.random.multivariate_normal(mean, sigma)
            sample = np.exp(sample) / np.sum(np.exp(sample))
            doc_topic_pro_list.append(sample)

    doc_topic_pro = np.array(doc_topic_pro_list)
    df_doc_topic = pd.DataFrame(doc_topic_pro, columns=topicnames, index=docnames)
    
    topic_word_pro = np.random.dirichlet(beta, num_topics)
    df_topic_word = pd.DataFrame(topic_word_pro, index=topicnames, columns=words)

    df_true_dist_list = [df_doc_topic, df_topic_word]

    docs = Parallel(n_jobs=num_jobs)(delayed(generate_doc)(i, 
        topicnames, 
        docnames, 
        words, 
        df_doc_topic, 
        df_topic_word, 
        min_words, 
        max_words) for i in tqdm(range(num_docs)))

    cov_names = ["cov_" + str(i) for i in range(num_covs)]
    temp = pd.DataFrame({'doc': docs,'doc_clean':docs})
    df = pd.concat([temp, pd.DataFrame(M_prevalence_covariates)], axis=1)

    df.columns = ["doc", "doc_clean"] + cov_names

    return df_true_dist_list, df