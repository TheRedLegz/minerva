import nltk

nltk.download('stopwords')
nltk.download('wordnet')

import numpy as np
import string as str
import pandas as pd
import math
import re
from pprint import pprint
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(string):
    
    def remove_numbers(string):
        res = re.sub(r'\d+', '', string)
        return res

    def remove_punctation(string):
        res = string.translate(string.maketrans("", "", str.punctuation))
        return res

    # NOTE this returns an array
    def remove_stop_words(string):
        split = string.split(' ')
        res = [w for w in split if w not in stop_words]
        return res

    def lemmatize_string(string):
        split = string

        if not isinstance(string, list):    
            split = string.split(' ')

        res = []

        for word in split:
            lemma = lemmatizer.lemmatize(word)
            res.append(lemma)

        return ' '.join(res)

    string = string.lower().strip()
    string = remove_numbers(string)
    string = remove_punctation(string)
    string = remove_stop_words(string)
    string = lemmatize_string(string)

    return string

def bag_of_words(document_array):

    doc_grams = []

    for doc in document_array:
        string = preprocess(doc)
        unigrams = nltk.word_tokenize(string)
        bigrams = nltk.bigrams(unigrams)
        bigrams = map(lambda x: x[0] + '_' + x[1], bigrams)
        uni_bi_grams = list(bigrams) + unigrams
        doc_grams.append(list(uni_bi_grams))


    unique_grams = []

    for doc in doc_grams:
        for bigram in doc:
            if bigram not in unique_grams:
                unique_grams.append(bigram)
    

    # doc = row, bigram = col
    document_count = len(doc_grams)
    gram_count = len(unique_grams)

    bow_grams = np.zeros((document_count, gram_count), dtype=int)
    

    for i in range(document_count):
        doc = doc_grams[i]

        for j in range(gram_count):
            gram = unique_grams[j]
            count = doc.count(gram)

            bow_grams[i][j] = int(count)

    return (bow_grams, unique_grams, doc_grams)


def tf_idf(document_array, bow = None):

    if bow is None:
        (bow, x, y) = bag_of_words(document_array)

    matrix = np.zeros(bow.shape, dtype=float)
    doc_count = len(document_array)

    for i, row in enumerate(bow):

        word_count = np.sum(row)

        for j, col in enumerate(row):
            
            tf = col / word_count
        
            has_word_count = np.count_nonzero(bow[:, j:j+1])
            idf = math.log(doc_count / (has_word_count + 1))

            matrix[i][j] = tf * idf

    return matrix

#--------------------MANUAL LSI--------------------
pi = math.pi


def SvD(Amatrix):
    '''
    Parameter:
        Amatrix - matrix to be decomposed of dimension m*n
    Process:
        Implements Jacobi method to find eigenvalues and eigenvectors.
        So, eigenvectors are in the V matrix. 
        S is a list of singular values of Amatrix. To find S, we just find square root 
        eigenvlues of Amatrix.
        We then find UMatrix using the formula U_i = (Amatrix*V_i)/Sigma_i 
    Output:
        U - Orthogonal matrix of dimension m*m 
        S - List of singular values of Amatrix
        VT - Transpose of orthogonal matrix V having dimensions n*n
    '''
    # Saves a copy of the original input matrix
    # This is done because the matrix is manipulated multiple times in the course of the method
    originalMatrix = Amatrix.copy()
    
    # NOTE: Perhaps change how this is implemented
    ROWS=len(Amatrix)
    COLS=len(Amatrix[0])

    # If ROWS is greater than COLS, it transposes the matrix
    # What this means is that the matrix is prepared in a way that the number of columns is greater than the rows
    #   i.e. A 3 x 10 matrix will not be transposed. However, a 10 x 3 matrix will be transposed to 3 x 10.
    #   i.e. cont. Other examples include: 7 x 2 -> 2 x 7; 8 x 5 -> 5 x 8;
    if (ROWS > COLS):
        Amatrix = np.transpose(Amatrix)
    B=Amatrix.copy()

    
    # Converts the matrix into a square matrix by multiplying the transposed matrix to itself
    #   i.e. 3 x 10 matrix -> (10 x 3)(3 x 10) = 10 x 10
    #   i.e. cont. Other examples include: 2 x 7 -> (7 x 2)(2 x 7) = 7 x 7; 5 x 8 -> (8 x 5)(5 x 8) = 8 x 8;
    Amatrix = np.matmul(np.transpose(Amatrix),Amatrix)

    # NOTE: Perhaps change the implementation here since it seems redundant
    # HERE Amatrix IS CONVERTED TO SQUARE MATRIX THUS ROWS=COLS
    ROWS=len(Amatrix)
    COLS=len(Amatrix[0])

    # Diagonal matrix is rounded to the nearest fifths place
    Amatrix=np.round(Amatrix,decimals=5)

    # A copy of the matrix is made
    # diagonalMatrix is then used for the following executions
    diagonalMatrix = Amatrix.copy()

    # Initializes an empty matrix with the size of the diagonal matrix
    # NOTE: Perhaps change implementation here. Seems redundant to use ROWS, ROWS when you can utilize the diagonalMatrix's shape
    eigenVectorsMatrix = np.zeros((ROWS,ROWS))
    for i in range(0,ROWS):
        eigenVectorsMatrix[i][i] = 1
    
    maximum=1

    # NOTE: Why not maximum > 0?
    # Loop is iterated untill the max element does not become 0.
    while(maximum>0.01):
        print("Max: ", maximum)
        # iOfMaxElement is the ith index of the max element other then diagonal
        # jOfMaxElement is the jth index of the max element other then diagonal
        
        # NOTE: Why start a [0][1]?
        maximum, iOfMaxElement,jOfMaxElement =math.fabs(diagonalMatrix[0][1]), 0, 1

        # Finds the location of the element with the highest value
        for i in range(0,len(diagonalMatrix)):
            for j in range(0,len(diagonalMatrix[0])):
                if(i!=j and math.fabs(diagonalMatrix[i][j]) > maximum):
                    maximum = math.fabs(diagonalMatrix[i][j])
                    iOfMaxElement = i
                    jOfMaxElement = j
        theta = 0

        # Finds the value of theta
        # Guide:
        #   - matrix[][] = the diagonal matrix of the input matrix
        #   - i = the i of the location of the maximal value of matrix[][]
        #   - j = is the j of the location of the maximal value of matrix[][]
        # if the matrix[i][i] has the same value as matrix[j][j] then...
        #   if matrix[i][j] is greater than 0 then...
        #       theta = pi/4
        #   else
        #       theta = -1(pi/4)
        # else
        #   theta = | 0.5 * (arctan((2 * matrix[i][j])/(matrix[i][i] - matrix[j][j]))) |

        if(diagonalMatrix[iOfMaxElement][iOfMaxElement] == diagonalMatrix[jOfMaxElement][jOfMaxElement]):
            if(diagonalMatrix[iOfMaxElement][jOfMaxElement] > 0):
                theta = pi/4
            else:
                theta = -1*pi/4
        else:
            value = 2*diagonalMatrix[iOfMaxElement][jOfMaxElement]/(diagonalMatrix[iOfMaxElement][iOfMaxElement] - diagonalMatrix[jOfMaxElement][jOfMaxElement])
            theta = abs(0.5*math.atan(value))


        # Initialize an empty matrix with the size of the square matrix of the input matrix
        #   i.e. input matrix 3 x 5 then OrthogonalMatrix = 5 x 5    
        # NOTE: Perhaps change np.zeroes to np.arange? Automatically initialize the matrix with 1s rather than 0s
        OrthogonalMatrix = np.zeros((len(diagonalMatrix),len(diagonalMatrix[0])))
        
        # Initializes the matrix's values to 1
        # See previous NOTE
        for i in range(0,len(diagonalMatrix)):
            OrthogonalMatrix[i][i] = 1

        # In the Orthogonal Matrix, here are the formulas for each manipulated value:
        # Guide:
        #   - matrix[][] = the diagonal matrix of the input matrix
        #   - oMatrix[][] = the orthogonal matrix
        #   - i = the i of the location of the maximal value of matrix[][]
        #   - j = is the j of the location of the maximal value of matrix[][]
        #   - theta = the value being calculated earlier (see above)
        # oMatrix[i][i] = cos(theta)
        # oMatrix[j][j] = oMatrix[i][i]
        # oMatrix[i][j] = sin(theta)
        # oMatrix[j][i] = -(oMatrix[i][j])
        # oMatrix[j][j] is a negative mirror of oMatrix[i][i]
        # oMatrix[j][i] is a negative mirror of oMatrix[i][j]
        # NOTE: I am not exactly sure what this is. Will do further research. !!! Search Jacobi Algorithm
        #   i = 0, j = 1
        #   k = cos(theta)
        #   l = sin(theta)
        #   k   l   x
        #  -l   k   x
        #   x   x   x
        OrthogonalMatrix[iOfMaxElement][iOfMaxElement] = math.cos(theta)
        OrthogonalMatrix[jOfMaxElement][jOfMaxElement] = OrthogonalMatrix[iOfMaxElement][iOfMaxElement] 
        OrthogonalMatrix[iOfMaxElement][jOfMaxElement] = math.sin(theta)
        OrthogonalMatrix[jOfMaxElement][iOfMaxElement] = -1*OrthogonalMatrix[iOfMaxElement][jOfMaxElement]

        
        #diagonalMatrix= P^(-1)*D*P     where P^(-1) is P transpose
        
        # 3 things happen here:
        # NOTE: n == columns of the AMatrix;
        #   - matrix = tranpose(oMatrix) * (matrix) == (n x n)(n x n) = n x n
        #   - matrix = matrix * oMatrix             == (n x n)(n x n) = n x n
        #   - eigenVectorsMatrix * oMatrix          == (n x n)(n x n) = n x n
        # NOTE: All matrices involved are n x n. Since they are all derived from the square matrix of the AMatrix

        # oMatrixT x diagonalMatrix x oMatrix
        diagonalMatrix = np.matmul(np.transpose(OrthogonalMatrix),diagonalMatrix)
        diagonalMatrix = np.matmul(diagonalMatrix,OrthogonalMatrix)
        
        # eigenVectorsMatrix x oMatrix
        eigenVectorsMatrix = np.matmul(eigenVectorsMatrix,OrthogonalMatrix)

        # End of loop
        # Simplified findings:
        #   It seems that the diagonal matrix is multiplied by the orthogonal matrix twice in each iteration. From what I can see, it sets up the the diagonal of the matrix to be in descending order.
        #   The eigenVectors depend on the orthogonal matrix. Since it is multiplied once, it is probably safe to assume that it's not a mirroring matrix
        
    # l1 is the list of eigen values which is extracted from the diagonal of the diagonalMatrix.
    l1 = []
    
    # Returns only the diagonals of the diagonalMatrix. The size of the list is n, where n == columns of the AMatrix
    for i in range(0,len(diagonalMatrix)):
        l1.append(diagonalMatrix[i][i])
    
    #eigenVectorsMatrix is a list of eigenvectors -- VT
    # NOTE: !!! This is apparently VT. Let's track a mental note here.
    eigenVectorsMatrix=np.transpose(eigenVectorsMatrix)

    # eigenVectors are flattened??? I am not sure what happens here. Perhaps a list of arrays? Must investigate
    tempList = list(eigenVectorsMatrix.copy())
    
    # Perhaps a list of lists. It seems as though it inserts the respective value of the diagonals to the head of each list.
    # NOTE: Perhaps change implementation. We can use Python's zip() method instead
    #   i.e
    #   l   x   x   x   x   x
    #   l   x   x   x   x   x
    #   l   x   x   x   x   x
    #   l   x   x   x   x   x
    #   l   x   x   x   x   x
    for i in range(0,len(tempList)):
        tempList[i] = list(tempList[i])
        tempList[i].insert(0,l1[i])
    
    #Sorting the eigen values in descending order and simuntaneously arranging the eigen vectors w.r.t descending eigen values.

    #First sorts the list based on the diagonals (or in this case the Eigen Values)
    tempList.sort(reverse = True)

    # Initializes the empty lists
    EigenValueslist=[]
    EigenFinalVectorslist=[]

    # Loops through tempList
    # NOTE: Perhaps change to dictionary?
    for i in range(0,len(tempList)):
        # First, we pop the value of the diagonals we previously inserted into the Eigen Values list.
        EigenValueslist.append(tempList[i].pop(0))
        # Second, we append the remaining elements of the list in the vectors list.
        EigenFinalVectorslist.append(tempList[i])

        # To sum it up, the point of appending then popping the value of the diagonals is to arrange them in descending order

    # These two converts the lists into np arrays
    EigenValuesarr = np.array(EigenValueslist)
    # VT is ready now
    # NOTE: !!! This is the VT
    EigenFinalVectorsTransarr=np.array(EigenFinalVectorslist)

    # Initiallizing U matrix with zeros (size m x m)
    # Keep in mind that B is the copy of the AMatrix BEFORE it was converted into a diagonal matrix.
    # NOTE: This is the U of A = U x Sigma x VT. So instead of it being n x n, where n == columns of A. It is m x m, where m == rows of A.
    UMatrix=np.zeros((len(B),len(B)))

    # Traverses the rows of the BMatrix (copy of AMatrix)
    for i in range(len(B)):
        # Ui=(B*Vi)/root(lamda)

        # As stated in the formula above, it multiplies the entire BMatrix with the corresponding i of the EigenVectorList
        # (m x n)(n x 1)
        # (m x 1)
        mul=np.matmul((B),(EigenFinalVectorsTransarr[i]))
        
        # NOTE: Not exactly sure what happens here. Perhaps divides each element with the square root of the Eigen Value (the diagonals from earlier)

        # x x x
        # y y y
        # ==
        # x/sqrt(y) x/sqrt(y) x/sqrt(y) 
        # (m x 1)

        UMatrix[i]=mul/math.sqrt(EigenValuesarr[i])
        # Iterates until it becomes (m x m)

    # Transpose the UMatrix
    # NOTE: I do not understand yet why
    UMatrix = np.transpose(UMatrix)

    # Initializes the final Eigen value list
    finEigVals = []

    # Traverses the first EigenValueArray
    for i in EigenValuesarr:
        # NOTE: Perhaps change implementation to just i != 0 ._.
        if(i > 1e-4): #we check if i!=0 
            finEigVals.append(i)
            
    # NOTE: !!!Singular values are square root of every eigen values.
    finSingVals = [math.sqrt(i) for i in finEigVals]
    # The values are rounded to the sixths place
    finSingVals = np.round(finSingVals,decimals = 6)

    # Initiallizing Sigma matrix = (m x n)
    # NOTE: The Sigma Matrix now uses the same size of the BMatrix
    Sigma = np.zeros((len(B),len(B[0])))
    
    # Plots the final singular values on to the diagonal of the Sigma matrix
    for i in range(0,min(len(B),len(B[0]))):
        Sigma[i][i] = finSingVals[i]

    VT = EigenFinalVectorsTransarr

    # Checks if the rows > cols (the same as the check in the first part of this method) and just transposes the formula
    # NOTE: Perhaps we can use a more elegant way of handling this check
    if(len(originalMatrix) > len(originalMatrix[0])):
        # For rows>columns we had taken Atranspose above and we had found U*Sigma*VT of AT
        # Thus this will be equivalent to V*SigmaT*UT of A.
        Sigma = np.transpose(Sigma)
        Utemp = UMatrix.copy()
        UMatrix = np.transpose(EigenFinalVectorsTransarr)
        VT = np.transpose(Utemp)

    # NOTE: Fixed the return values from finSingValues to Sigma. Perhaps the dev wants to return a list instead?
    # return UMatrix,finSingVals,VT
    return UMatrix, Sigma, VT

#--------------------LSI ENDS HERE--------------------




# DRIVER HERE



raw = pd.read_json('sample.json')
data = []

for i, row in raw.iterrows():
    data.append(row['full_text'])

tf_idf_data = tf_idf(data)
pprint(tf_idf_data)
pprint(tf_idf_data.shape)
(U, S, VT) = SvD(tf_idf_data)
pprint(U.shape)
pprint(S.shape)
pprint(VT.shape)
pprint(VT)