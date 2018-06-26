import numpy as np
import matplotlib.pyplot as plt  
from scipy.io import loadmat
from scipy.optimize import minimize
 
 
data = loadmat('data/ex8_movies.mat')
print(data.keys())
Y = data['Y'] # scores 1 to 5 (0 if movie wasn't rated by the particular user)
R = data['R'] # 0 - user didn't rate the movie, 1 - user rated the movie

print("(number of movies, number of users):", Y.shape)
print("(number of movies, number of users):", R.shape)

# print(Y[0:10, :])
# print(R[0:10, :])

avg_first_movie = Y[0, np.where(R[0, :] == 1)[0]].mean()
print("avg for the first movie:", avg_first_movie)

plt.imshow(Y)
plt.show()


# the cost function is a difference between true ratings of a movies
# and predicted ratings
def cost(params, Y, R, num_features, learning_rate):
    # the cost fuction is based on two sets of parameters: X and Theta
    # that are passed to the function as params argument
    Y = np.matrix(Y)  # (1682, 943)
    R = np.matrix(R)  # (1682, 943)

    num_movies = Y.shape[0]
    num_users = Y.shape[1]

    # reshape the parameter array into parameter matrices
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))  # (1682, 10)
    Theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))  # (943, 10)

    # initialization:
    J = 0
    error = np.zeros((num_movies, num_users))
    X_grad = np.zeros(X.shape)  # (1682, 10)
    Theta_grad = np.zeros(Theta.shape)  # (943, 10)
    # compute the cost
    # j - index of user
    # i - index of movie
    for j in range(num_users):
        for i in range(num_movies):
            if R[i, j] != 0:
                error[i, j] = (Theta[j, :] * X[i, :].T) - Y[i, j]
                squared_error = np.power(error[i, j], 2)
                J += squared_error
    J *= 0.5

    # add the cost regularization
    J = J + learning_rate/2 * np.sum(np.power(Theta, 2))
    J = J + learning_rate/2 * np.sum(np.power(X, 2))

    # calculate the gradients with regularization
    X_grad = (error * Theta) + (learning_rate * X)
    Theta_grad = (error.T * X) + (learning_rate * Theta)

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))

    return np.ravel(J)[0], grad

# predict how user will rate a movie X
users = 4
movies = 5
features = 3

#load parameters
params_data = loadmat('data/ex8_movieParams.mat')
X = params_data['X'] # contains measurements of features (e.g. how romantic it is) for each movie
Theta = params_data['Theta'] # contains coefficients that allows to predict user's rate of a movie

X_sub = X[:movies, :features] # num_movies x num_features
Theta_sub = Theta[:users, :features] # num_users x num_features
Y_sub = Y[:movies, :users]
R_sub = R[:movies, :users]
params = np.concatenate((np.ravel(X_sub), np.ravel(Theta_sub)))

J, grad = cost(params, Y_sub, R_sub, features, 1.5)
print("Cost function J:", J, "\ngrad:", grad)

movie_idx = {}
f = open('data/movie_ids.txt')
for line in f:
    tokens = line.split(' ')
    tokens[-1] = tokens[-1][:-1]
    movie_idx[int(tokens[0]) - 1] = ' '.join(tokens[1:])

ratings = np.zeros((1682, 1))
ratings[0] = 1
ratings[2] = 5
ratings[13] = 2
ratings[51] = 4
ratings[59] = 3
ratings[62] = 2
ratings[101] = 5
ratings[103] = 1
ratings[192] = 1
ratings[245] = 4
ratings[332] = 5

print('Rated {0} with {1} stars.'.format(movie_idx[0], str(int(ratings[0]))))
print('Rated {0} with {1} stars.'.format(movie_idx[2], str(int(ratings[2]))))
print('Rated {0} with {1} stars.'.format(movie_idx[13], str(int(ratings[13]))))
print('Rated {0} with {1} stars.'.format(movie_idx[51], str(int(ratings[51]))))
print('Rated {0} with {1} stars.'.format(movie_idx[59], str(int(ratings[59]))))
print('Rated {0} with {1} stars.'.format(movie_idx[62], str(int(ratings[62]))))
print('Rated {0} with {1} stars.'.format(movie_idx[101], str(int(ratings[101]))))
print('Rated {0} with {1} stars.'.format(movie_idx[103], str(int(ratings[103]))))
print('Rated {0} with {1} stars.'.format(movie_idx[192], str(int(ratings[192]))))
print('Rated {0} with {1} stars.'.format(movie_idx[245], str(int(ratings[245]))))
print('Rated {0} with {1} stars.'.format(movie_idx[332], str(int(ratings[332]))))

R = data['R']
Y = data['Y']

Y = np.append(Y, ratings, axis=1)
R = np.append(R, ratings != 0, axis=1)

movies = Y.shape[0]
users = Y.shape[1]
features = 10
learning_rate = 10.

X = np.random.random(size=(movies, features))
Theta = np.random.random(size=(users, features))
params = np.concatenate((np.ravel(X), np.ravel(Theta)))

Ymean = np.zeros((movies, 1))
Ynorm = np.zeros((movies, users))

for i in range(movies):
    idx = np.where(R[i, :] == 1)[0]
    Ymean[i] = Y[i, idx].mean()
    Ynorm[i, idx] = Y[i, idx] - Ymean[i]
print("here")
fmin = minimize(fun=cost, x0=params, args=(Ynorm, R, features, learning_rate),
                method='CG', jac=True, options={'maxiter': 100})

print("fmin:", fmin)

X = np.matrix(np.reshape(fmin.x[:movies * features], (movies, features)))
Theta = np.matrix(np.reshape(fmin.x[movies * features:], (users, features)))

predictions = X * Theta.T
my_preds = predictions[:, -1] + Ymean
sorted_preds = np.sort(my_preds, axis=0)[::-1]
sorted_preds[:10]

idx = np.argsort(my_preds, axis=0)[::-1] # (use np.argsort() to get the movie index for ratings)
print("Top 10 movie predictions:")
for i in range(10):
    j = int(idx[i])
    print('Predicted rating of {0} for movie {1}.'.format(str(float(my_preds[j])), movie_idx[j]))

