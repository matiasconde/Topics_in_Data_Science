#Natural language processing is the study of enabling computers to understand human languages. This field may involve teaching computers to #automatically score essays, infer grammatical rules, or determine the emotions associated with text.

#In order for a computer to begin making inferences from a given text, we'll need to convert the text to a numerical representation. This #process will enable the computer to intuit grammatical rules, which is more akin to learning a first language.

#One technique is put all the words from the text aligned , then for every sentence, we put a 1 if the word occur and a 0 if the word don't #occur

#The complete text has a length of n, then every sentence has the same length, but with 0's and 1's whenever the word appears on the #sentence or not.

#Bag of words Model (it does not matter the order of the words, nor the grammar, it only matters the multiplicity or the number of times #that the words appear in the text)

#1. First Step , tokenize

tokenized_headlines = []

for item in submissions["headline"]:
    tokens_row = item.split()
    tokenized_headlines.append(tokens_row)

#2. Second Step, proceesing text: Removing punctuation and lowercase the tokens.

punctuation = [",", ":", ";", ".", "'", '"', "’", "?", "/", "-", "+", "&", "(", ")"]
clean_tokenized = []

for sentence in tokenized_headlines: 
    clean_sentence = []
    for token in sentence: 
        lower_token = token.lower()
        for item in punctuation: 
            if item in lower_token:
                lower_token = lower_token.replace(item,"")
        clean_sentence.append(lower_token)
    clean_tokenized.append(clean_sentence)

#3. Third Step, identify unique words (those that's only appears one time doesn't contribute to the model)
#then creating a Matrix with all the words as column names and an amount of rows as the amunt of sentences. 
#Every row will be a chain of 1's and 0's marking the sentence. 

# First create a DataFrame full of zeros with a size like we did explain:
import numpy as np
import pandas as pd
unique_tokens = []
single_tokens = []

count_words = {}

for sentence in clean_tokenized: 
    for word in sentence: 
        if word not in count_words:
            count_words[word] = 1
        else: count_words[word] += 1

for k,v in count_words.items():
    if v==1: single_tokens.append(k)

for sentence in clean_tokenized: 
    for word in sentence: 
        if word not in unique_tokens and word not in single_tokens:
            unique_tokens.append(word)
        
counts = pd.DataFrame(0,index=np.arange(len(clean_tokenized)),columns = unique_tokens)

# Now let's fill the appearances of the words.
for i,sentence in enumerate(clean_tokenized):
    for word in sentence:
        if word in counts.columns:
             counts.loc[i,word] += 1

#4. Fourth Step: Removing features (or words) that overfit and cause noise (stopwords, and "low-appears" words)
#To keeping simpler, we are going to remove words that appear less than 5 and more than 100 times: 

word_counts = counts.sum(axis=0)

filter_words = (5<=word_counts)&(word_counts<=100)

counts = counts.loc[:,filter_words]

#5. Five Step: splitting data-set in train-test:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(counts, submissions["upvotes"], test_size=0.2, random_state=1)

#5. Fiveth Step: fitting and predicting, first with LinearRegression

from sklearn.linear_model import LinearRegression

clf = LinearRegression()

clf.fit(X_train,y_train)

predictions = clf.predict(X_test)

squares = [(predictions[i] - v)**2 for i,v in enumerate(y_test)]
    
mse = sum(squares) / len(y_test)

#Add "meta" features like headline length and average word length.
#Use a random forest, or another more powerful machine learning technique.
#Explore different thresholds for removing extraneous columns.

Download the full data in 
https://github.com/arnauddri/hn/blob/master/data/comments.7z.001
