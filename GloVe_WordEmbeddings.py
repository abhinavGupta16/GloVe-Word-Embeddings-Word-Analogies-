from scipy import spatial
from scipy.stats import pearsonr
import numpy as np
import argparse

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('-f', '--gloveFilePath', help='Glove File name', required=True)
	parser.add_argument('-w', '--wordSim', help='Run Word Sim', action='store_true', default=False)
	parser.add_argument('-a', '--analogy', help='Run Analogy', action='store_true', default=False)

	args = parser.parse_args()

	if args.wordSim:
		wordsim(args.gloveFilePath)

	if args.analogy:
		print("Finding Analogies : ")
		analogy(args.gloveFilePath)


def analogy(gloveFilePath):
	gloveFileDict = readGloveFile(gloveFilePath)
	output_file = open("analogy_output/" + gloveFilePath.split(".txt")[0].replace("/", "_") + "_output.txt", 'w+')
	output_file.write("Computed analogies on RHS - \n")
	with open("analogy.txt") as infile:
		for line in infile:
			if line.strip() == "":
				continue
			line = line.rstrip('\n')
			arr = line.split(" ")

			wordVec1 = gloveFileDict[arr[0]]
			wordVec2 = gloveFileDict[arr[1]]
			wordVec3 = gloveFileDict[arr[2]]

			finalWordVec = np.add(np.subtract(wordVec1, wordVec2), wordVec3)

			# print("wordVec1 : " , wordVec1)
			# print("wordVec2 : " , wordVec2)
			# print("wordVec3 : " , wordVec3)
			# print("wordVecfinal : " , finalWordVec)

			similarWord = findSimilarWord(arr[0], arr[1], arr[2], finalWordVec, gloveFileDict)

			output_file.write(arr[0] + " - " + arr[1] + " + " + arr[2] + " = " + similarWord + "\n")
			print(arr[0] + " - " + arr[1] + " + " + arr[2] + " = " + similarWord)
	output_file.close()


def findSimilarWord(word1, word2, word3, wordVec, gloveFileDict):
	maxCosWord = ""
	maxCosineVal = 0.0
	for key, value in gloveFileDict.items():
		word = key
		result = 1 - spatial.distance.cosine(wordVec, value)
		result = round(result, 3)
		if result > maxCosineVal and (word1 != word and word2 != word and word3 != word):
			maxCosineVal = result
			maxCosWord = word

	return maxCosWord

def wordsim(gloveFilePath):
	gloveFileDict = readGloveFile(gloveFilePath)
	output_file = open("output/" + gloveFilePath.split(".txt")[0].replace("/","_") + "_output.txt", 'w+')
	humanScoreArr = []
	cosineScoreArr = []

	with open("wordsim-353.txt") as infile:
		for line in infile:
			line = line.strip()
			if line == "" or line[0] == '#':
				continue
			line = line.rstrip('\n')
			arr = line.split('\t')
			arr.pop(0)
			word1 = arr.pop(0).strip()
			word2 = arr.pop(0).strip()
			humanScore = float(arr.pop(0))
			humanScoreArr.append(humanScore)
			word1Vec = gloveFileDict[word1.lower()]
			word2Vec = gloveFileDict[word2.lower()]
			result = 1 - spatial.distance.cosine(word1Vec, word2Vec)
			# result = dot(word1Vec, word2Vec) / (norm(word1Vec) * norm(word2Vec))
			result = round(result, 2)
			cosineScoreArr.append(result)
			output_file.write(word1 + '\t' + word2 + '\t' + str(humanScore) + '\t' + str(result) + '\n')

	correlation, p_value = pearsonr(humanScoreArr, cosineScoreArr)

	output_file.write("\n \ncorrelation: " + str(correlation) + "\n" + "p_value: " + str(p_value) + "\n")

	print("pearson ", correlation, p_value)
	output_file.close()
	infile.close()


def readGloveFile(filePath):
	gloveFileDict = {}
	with open(filePath) as infile:
		for line in infile:
			if line.strip() == "":
				continue
			arr = line.split(' ')
			word = arr.pop(0).strip()
			gloveFileDict[word.lower()] = list(map(float, arr))
	infile.close()
	return gloveFileDict

if __name__ == "__main__":
	main()