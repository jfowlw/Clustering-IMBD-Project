import csv
import codecs
import operator
import numpy as np
import sys
from random import randint
import random

budget = []
genres = []
keywords = []
movieid = []
original_language = []
popularity = []
production_companies = []
production_countries = []
revenue = []
runtime = []
spoken_languages = []
vote_average = []
vote_count = []

newRuntimes = []
vectorGenres = []
vectorKeywords = []

budgCentroids = []
popCentroids = []
revCentroids = []
genCentroids = []
keyCentroids = []

total_votes = []

def getFile(path):
    enc = 'utf-8'

    with codecs.open(path, encoding=enc, errors='ignore') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        next(csvfile)

        for row in readCSV:
            budget.append(float(row[0]))
            genres.append(row[1])
            movieid.append(int(row[3]))
            keywords.append(row[4])
            original_language.append(row[5])
            popularity.append(float(row[8]))
            production_companies.append(row[9])
            production_countries.append(row[10])
            revenue.append(float(row[12]))
            runtime.append((row[13]))
            spoken_languages.append(row[14])
            vote_average.append(float(row[18]))
            vote_count.append(float(row[19]))

def getGenres():
    concatGenres=[]

    for genre in genres:
        temp1=genre.replace('[', '')
        concatGenre=temp1.replace(']', '')
        concatGenres.append(concatGenre)

    b=[]
    for genre in concatGenres:
        b.append(genre.split(", "))

    rankGenres=[]
    for i in b:
        for a in i:
            if "id" not in a:
                rankGenres.append(a)
    freq={i:rankGenres.count(i) for i in set(rankGenres)}
    sort = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)
    top_genres=[]
    for a in sort[0:20]:
        top_genres.append(a[0])
    return top_genres

def getKeywords():
    concatKeywords=[]
    for keyword in keywords:
        temp1=keyword.replace('[', '')
        concatKeyword=temp1.replace(']', '')
        concatKeywords.append(concatKeyword)

    b=[]
    for keyword in concatKeywords:
        b.append(keyword.split(", "))

    rankKeywords=[]
    for i in b:
        for a in i:
            if "id" not in a:
                rankKeywords.append(a)
    freq={i:rankKeywords.count(i) for i in set(rankKeywords)}
    sort = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)
    top_keywords=[]
    for a in sort[0:20]:
        top_keywords.append(a[0])
    return top_keywords

def vectorizeGenres(checks):
    for genre in genres:
        temp = ""
        for check in checks:
            if check in genre:
                temp = temp + "1"
            else:
                temp = temp + "0"
        vectorGenres.append(temp)

def vectorizeKeywords(checks):
    for keyword in keywords:
        temp = ""
        for check in checks:
            if check in keyword:
                temp = temp + "1"
            else:
                temp = temp + "0"
        vectorKeywords.append(temp)

def vectorDistance(index, centroid):
    words1 = zip(vectorGenres[index], genCentroids[centroid])
    incorrect1 = len([c for c, d in words1 if c != d])
    words2 = zip(vectorKeywords[index], keyCentroids[centroid])
    incorrect2 = len([c for c, d in words2 if c != d])
    return ((incorrect1)**2) + ((incorrect2)**2)

def intDistance(index, centroid):
    budgDistance = pow((budget[index]-budgCentroids[centroid]), 2)
    popDistance = pow((popularity[index]-popCentroids[centroid]), 2)
    revDistance = pow((revenue[index]-revCentroids[centroid]), 2)
    return budgDistance + popDistance + revDistance

def vectorCompare(a, b):
    words = zip(a, b)
    incorrect = len([c for c, d in words if c != d])
    return (incorrect/len(a))**2

def kmeans(data, k, cutoff, max_it, plus):
    centroids = []

    #Randomly select centers
    centerIndex = []
    for i in range(k):
        index = randint(0, len(data))
        while (index in centerIndex or data[index] in centroids):
            index = randint(0, len(data))
        centroids.append(i)
        budgCentroids.append(budget[index])
        popCentroids.append(popularity[index])
        revCentroids.append(revenue[index])
        genCentroids.append(vectorGenres[index])
        keyCentroids.append(vectorKeywords[index])
        centerIndex.append(index)

    #Run k-means++ if necessary
    if(plus):
        classes = {}
        distances = {}
        for i in range(k):
            classes[i] = []
            distances[i] = []
        # Find the nearest centroid for each point
        for index, point in enumerate(data):
            minDistance = float('inf')
            minIndex = 0
            for centroid in centroids:
                distance = intDistance(index, centroid) + vectorDistance(index, centroid)
                if distance < minDistance:
                    minDistance = distance
                    minIndex = centroids.index(centroid)
            classes[minIndex].append(index)
            distances[minIndex].append(minDistance)

        # Find new centroids based on distances array
        for index, cluster in enumerate(distances):
            # Find probabilities for each cluster
            r = random.random()
            sum = 0.0
            cumprobs = 0.0
            for distance in distances[cluster]:
                sum = sum + distance
            for ind, distance in enumerate(distances[cluster]):
                probs = distance/sum
                cumprobs = cumprobs + probs
                if cumprobs > r:
                    budgCentroids[index] = budget[classes[index][ind]]
                    popCentroids[index] = popularity[classes[index][ind]]
                    revCentroids[index] = revenue[classes[index][ind]]
                    genCentroids[index] = vectorGenres[classes[index][ind]]
                    keyCentroids[index] = vectorKeywords[classes[index][ind]]

    #Loop to find optimal centers
    for i in range(max_it):
        classes = {}
        for j in range(k):
            classes[j] = []

        #Find the nearest centroid for each point
        for index, point in enumerate(data):
            minDistance = float('inf')
            minIndex = 0
            for centroid in centroids:
                distance = intDistance(index, centroid) + vectorDistance(index, centroid)
                if distance < minDistance:
                    minDistance = distance
                    minIndex = centroids.index(centroid)
            classes[minIndex].append(index)

        previousBudg = budgCentroids[:]
        previousPop = popCentroids[:]
        previousRev = revCentroids[:]
        previousGen = genCentroids[:]
        previousKey = keyCentroids[:]

        #Average the points in each cluster to recalculate the centroids
        for classification in classes:
            #Averages for budget, popularity, and revenue
            budgSum = 0
            popSum = 0
            revSum = 0
            for element in classes[classification]:
                budgSum = budgSum + budget[element]
                popSum = popSum + popularity[element]
                revSum = revSum + revenue[element]
            budgCentroids[classification] = budgSum / (len(classes[classification]) + 1)
            popCentroids[classification] = popSum / (len(classes[classification]) + 1)
            revCentroids[classification] = revSum / (len(classes[classification]) + 1)
            #Averages for genre and keywords
            genSum = ""
            for j in range(len(genCentroids[0])):
                count = 0
                for element in classes[classification]:
                    if vectorGenres[element][j] == "1":
                        count = count + 1
                if count > (len(genCentroids[0]) / 2):
                    genSum = genSum + "1"
                else:
                    genSum = genSum + "0"
            genCentroids[classification] = genSum
            keySum = ""
            for j in range(len(keyCentroids[0])):
                count = 0
                for element in classes[classification]:
                    if vectorKeywords[element][j] == "1":
                        count = count + 1
                if count > (len(keyCentroids[0]) / 2):
                    keySum = keySum + "1"
                else:
                    keySum = keySum + "0"
            keyCentroids[classification] = keySum


        #Check the difference from previous clustering
        isOptimal = True

        sum = 0
        for j in range(k):
            budgDif = pow((previousBudg[j] - budgCentroids[j]), 2)
            popDif = pow((previousPop[j] - popCentroids[j]), 2)
            revDif = pow((previousRev[j] - revCentroids[j]), 2)
            genDif = vectorCompare(previousGen[j], genCentroids[j])
            keyDif = vectorCompare(previousKey[j], keyCentroids[j])
            sum = budgDif + popDif + revDif + genDif + keyDif

        if sum > cutoff:
            isOptimal = False

        if isOptimal:
            return classes;

    return classes;

#Main Source Code
def main():
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    arg3 = sys.argv[3]
    getFile(arg1)
    k = int(arg2)
    cutoff = 10**(-2)
    max_it = 1000
    if arg3 == "random":
        plus = False
    else:
        plus = True

    top_genres = getGenres()
    top_keywords = getKeywords()

    vectorizeGenres(top_genres)
    vectorizeKeywords(top_keywords)

    clusters = kmeans(budget, k, cutoff, max_it, plus)

    filename = "output.csv"
    headers = "id, label\n"
    f = open(filename, "w")
    f.write(headers)
    for index, cluster in enumerate(clusters):
        for element in clusters[cluster]:
            f.write(str(movieid[element]) + " , " + str(index) + "\n")

    f.close()

main()

