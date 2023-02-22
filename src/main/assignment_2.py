import numpy as np

np.set_printoptions(precision=7, suppress=True, linewidth=100)

def nevilles_method(x_points, y_points, x):
    matrix = np.zeros((len(y_points), len(y_points)))
    
    # fillign in y values
    for counter, row in enumerate(matrix):
        row[0] = y_points[counter]
    
    # populating the matrix
    for i in range(1, len(x_points)):
        for j in range(1, i+1):
            first_multiplication = (x - x_points[i - j]) * matrix[i][j-1]
            second_multiplication = (x - x_points[i]) * matrix[i-1][j-1]
            denominator = x_points[i] - x_points[i-j]
            # this is the value that we will find in the matrix
            coefficient = (first_multiplication - second_multiplication) / denominator
            matrix[i][j] = coefficient
            
    print(matrix[len(y_points) - 1, len(y_points) - 1], '\n')


def divided_difference_table(x_points, y_points):
    # setting up the matrix
    size = len(y_points)
    matrix = np.zeros((size, size))
    
    # filling the matrix
    for counter, row in enumerate(matrix):
        row[0] = y_points[counter]
        
    # populating the matrix
    for i in range(1, size):
        for j in range(1, i+1):
            
            numerator = matrix[i][j-1] - matrix[i-1][j-1]
            
            denominator = x_points[i] - x_points[i-j]
            operation = numerator / denominator
            
            matrix[i][j] = operation

    approxVector = [0] * (size - 1)
    
    for i in range(0, size - 1):
        approxVector[i] = matrix[i+1][i+1]
    
    print(approxVector, '\n')
    return matrix

def approximateValue(matrix, x_points, value):
    reoccuring_x_span = 1
    reoccuring_px_result = matrix[0,0]
    
    for index in range(1, len(x_points)):
        polynomial_coefficient = matrix[index][index]
        
        reoccuring_x_span *= (value - x_points[index - 1])
        
        mult_operation = polynomial_coefficient * reoccuring_x_span
        
        reoccuring_px_result += mult_operation
    
    print(reoccuring_px_result, '\n')


def apply_div_dif(matrix: np.array):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i+2):
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue
            
            left: float = matrix[i][j-1]
            
            diagonal_left: float = matrix[i-1][j-1]
            
            numerator: float = left - diagonal_left
            
            denominator = matrix[i][0] - matrix[i - (j-1)][0]
            
            operation = numerator / denominator
            matrix[i][j] = operation
    return matrix
    

def hermite_interpolation(x_points, y_points, slopes):
    
    num_of_points = len(x_points)
    matrix = np.zeros((num_of_points * 2, num_of_points * 2 + 1))
    
    index = 0
    for x in range(0, num_of_points * 2):
        if (x % 2) == 0:
            matrix[x][0] = x_points[index]
            matrix[x][1] = y_points[index]
        else:
            matrix[x][0] = x_points[index]
            matrix[x][1] = y_points[index]
            index += 1
 
    index = 0
    for x in range(1, num_of_points * 2):
        if (x % 2) != 0:
            matrix[x][2] = slopes[index]
            index += 1
    filled_matrix = apply_div_dif(matrix)
    print(filled_matrix, '\n')

def cubicSplineInterpolation(x_points, y_points):
    size = len(y_points)
    matrix = np.zeros((size, size))
    
    matrix[0][0] = 1
    matrix[size-1][size-1] = 1
    
    # setting up h_0
    matrix[1,0] = x_points[1] - x_points[0]
    
    for i in range(1, size - 1):
        h_i = x_points[i+1] - x_points[i]
        
        matrix[i+1][i] = h_i
        matrix[i][i+1] = h_i
        matrix[i][i] = 2 * (matrix[i][i-1] + h_i)
    
    matrix[size-1][size-2] = 0
    print(matrix, '\n')
    
    # setting up empty array for b
    b = [0] * (size)
    
    for i in range(1, size - 1):
        firstHalf = (3 / (x_points[i+1] - x_points[i])) * (y_points[i+1] - y_points[i])
        secondHalf = (3 / (x_points[i] - x_points[i-1])) * (y_points[i] - y_points[i-1])
        b[i] = firstHalf - secondHalf
    
    b = np.array(b)
    print(b, '\n')
    
    x = np.linalg.solve(matrix, b)
    print(x, '\n')

if __name__ == "__main__":
    
    x_points= [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    x = 3.7
    nevilles_method(x_points, y_points, x)
    
    x_points = [7.2, 7.4, 7.5, 7.6]
    y_points = [23.5492, 25.3913, 26.8224, 27.4589]
    m = divided_difference_table(x_points, y_points)
    
    approximateValue(m, x_points, 7.3)
    
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    slopes = [-1.195, -1.188, -1.182]
    hermite_interpolation(x_points, y_points, slopes)
    
    x_points = [0,1,2,3]
    y_points = [1,2,4,8]
    x_points = [2, 5, 8, 10]
    y_points = [3, 5, 7, 9]
    cubicSplineInterpolation(x_points, y_points)