from operator import index
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


tweets_df = pd.read_json (r'sample.json')

def preprocessing():
    stop_words = stopwords.words('english')
    tweets_df['clean_doc'] = tweets_df['full_text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
    tweets_df['clean_doc'] = tweets_df['clean_doc'].str.replace("[^a-zA-Z#]", " ", regex=True)
    tweets_df['clean_doc'] = tweets_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    tweets_df['clean_doc'] = tweets_df['clean_doc'].apply(lambda x: x.lower())
    tokenized_doc = tweets_df['clean_doc'].apply(lambda x: x.split())
    tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
    detokenized_doc = []
    for i in range(len(tweets_df)):
        t = ' '.join(tokenized_doc[i])
        detokenized_doc.append(t)

    tweets_df['clean_doc'] = detokenized_doc

def topicModeling():
    vectorizer = TfidfVectorizer(stop_words='english', 
    max_features= 1000, # keep top 1000 terms 
    max_df = 0.5, 
    smooth_idf=True)
    X = vectorizer.fit_transform(tweets_df['clean_doc'])
    # print(X.shape)

    svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)
    svd_model.fit(X)
    terms = vectorizer.get_feature_names()

    # print(len(terms))
    # for i, comp in enumerate(svd_model.components_):
    #     terms_comp = zip(terms, comp)
    #     sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
        # print("Topic "+str(i)+": ")
        # for t in sorted_terms:
        #     print(t)
        #     print(" ")
        # print(sorted_terms)

    pca = PCA(n_components = 2)
    principalComponents = pca.fit_transform(svd_model.components_)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['1', '2'])
    
    print(principalDf)

#--------------------MANUAL LSI--------------------
pi = math.pi

# NOTE: Redundant. Use np library.
def matmul(A,X):
    '''
    Parameter: 
        A - 2D matrix having dimension m*r
        X - 2D matrix having dimension r*n
    Process:
        matrix multiplication of A and X
    Output:
        ans - 2D matrix having dimension m*n
    '''
    
    if(type(X)!=list and len(X.shape)==1):
        X = X.reshape((len(X), 1))
    ans = np.zeros((len(A),len(X[0])))
    for i in range(len(A)):
        for j in range(len(X[0])):
            for k in range(len(X)):
                ans[i][j] = ans[i][j] + A[i][k]*X[k][j]
    if(type(X)!=list and ans.shape[1]==1):
        ans = ans.reshape((len(ans)))
    return ans

# NOTE: Redundant. Use np library.
def transpose(A):
    '''
    Parameter: 
        A - 2D matrix having dimension m*n
    Process:
        Calculates transpose of given matrix
    Output:
        ans - 2D matrix having dimension n*m
    '''
    ans = np.zeros((len(A[0]),len(A)))
    for i in range(len(A)):
        for j in range(len(A[0])):
            ans[j][i] = A[i][j]
    return ans


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
        Amatrix = transpose(Amatrix)
    B=Amatrix.copy()

    
    # Converts the matrix into a square matrix by multiplying the transposed matrix to itself
    #   i.e. 3 x 10 matrix -> (10 x 3)(3 x 10) = 10 x 10
    #   i.e. cont. Other examples include: 2 x 7 -> (7 x 2)(2 x 7) = 7 x 7; 5 x 8 -> (8 x 5)(5 x 8) = 8 x 8;
    Amatrix = matmul(transpose(Amatrix),Amatrix)

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

    # Loop is iterated untill the max element does not become 0.
    while(maximum>0.001):
        # iOfMaxElement is the ith index of the max element other then diagonal
        # jOfMaxElement is the jth index of the max element other then diagonal
        
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

        if(diagonalMatrix[iOfMaxElement][iOfMaxElement] == diagonalMatrix[jOfMaxElement][jOfMaxElement]):
            if(diagonalMatrix[iOfMaxElement][jOfMaxElement] > 0):
                theta = pi/4
            else:
                theta = -1*pi/4
        else:
            value = 2*diagonalMatrix[iOfMaxElement][jOfMaxElement]/(diagonalMatrix[iOfMaxElement][iOfMaxElement] - diagonalMatrix[jOfMaxElement][jOfMaxElement])
            theta = abs(0.5*math.atan(value))


        # Initialize an empty matrix with the size of the square matrix of the input matrix
        #   i.e. input matrix 3 x 5 then OrthogonalMatrix = 3 x 3    
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

        diagonalMatrix = matmul(transpose(OrthogonalMatrix),diagonalMatrix)
        
        diagonalMatrix = matmul(diagonalMatrix,OrthogonalMatrix)
        
        eigenVectorsMatrix = matmul(eigenVectorsMatrix,OrthogonalMatrix)

        # End of loop
        # Simplified findings:
        #   It seems that the diagonal matrix is multiplied by the orthogonal matrix twice in each iteration. From what I can see, it sets up the the diagonal of the matrix to be in descending order.
        #   The eigenVectors depend on the orthogonal matrix. Since it is multiplied once, it is probably safe to assume that it's not a mirroring matrix
        
    # l1 is the list of eigen values which is extracted from the diagonal of the diagonalMatrix.
    l1 = []
    
    # Returns only the diagonals of the diagonalMatrix. The size of the list is m, where m == rows of the AMatrix
    for i in range(0,len(diagonalMatrix)):
        l1.append(diagonalMatrix[i][i])
    
    #eigenVectorsMatrix is a list of eigenvectors -- VT
    # NOTE: !!! This is apparently VT. Let's track a mental note here.
    eigenVectorsMatrix=transpose(eigenVectorsMatrix)

    # eigenVectors are flattened??? I am not sure what happens here. Perhaps a list of arrays? Must investigate
    tempList = list(eigenVectorsMatrix.copy())
    
    # Perhaps a list of lists. It seems as though it inserts the respective value of the diagonals to the head of each list.
    # NOTE: Perhaps change implementation. We can use Python's zip() method instead
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
        mul=matmul((B),(EigenFinalVectorsTransarr[i]))
        
        # NOTE: Not exactly sure what happens here. Perhaps divides each element with the square root of the Eigen Value (the diagonals from earlier)
        UMatrix[i]=mul/math.sqrt(EigenValuesarr[i])

    # Transpose the UMatrix
    # NOTE: I do not understand yet why
    UMatrix = transpose(UMatrix)

    # Initializes the final Eigen value list
    finEigVals = []

    # Traverses the first EigenValueArray
    for i in EigenValuesarr:
        # NOTE: Perhaps change implementation to just i != 0 ._.
        if(i > 1e-4): #we check if i!=0 
            finEigVals.append(i)
            
    # NOTE: !!!Singular values are root of every eigen values.
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
        Sigma = transpose(Sigma)
        Utemp = UMatrix.copy()
        UMatrix = transpose(EigenFinalVectorsTransarr)
        VT = transpose(Utemp)

    # NOTE: Fixed the return values from finSingValues to Sigma
    # return UMatrix,finSingVals,VT
    return UMatrix, Sigma, VT

#--------------------LSI ENDS HERE--------------------

preprocessing()
topicModeling()
