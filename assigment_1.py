#Import Statements.
import numpy as np
import matplotlib.pyplot as plt 


    
# Import Statements.
import numpy as np
import matplotlib.pyplot as plt


def file_name():
    '''
     Create function file_name.
     It will return the dataset of salaries in europe.

     Returns
     -------
     salary_file : numpy array
         Numpy array containing datset of salaries in europe.
     ohist : variable
         histogram function.
     oedge : variable
         histogram function.

     '''
    # Import CSV file.
    salary_file = np.genfromtxt("data0-1.csv", delimiter=',')
    ohist, oedge = np.histogram(salary_file, bins=36)

    return salary_file, ohist, oedge


def bin_cent():
    '''
    This function will return the calculated bin center and bin location.

    Returns
    -------
    x_dist : variable
        Center of bins.
    w_dist : variable
        Width of bins.
    y_dist : variable
        probability distribution.

    '''
    # Fist we will calculate the bin center and bin location.
    x_dist = 0.5 * (oedge[1:] + oedge[:-1])
    w_dist = oedge[1:] - oedge[:-1]
    # Normalise the distribution
    y_dist = ohist / np.sum(ohist)
    return x_dist, w_dist, y_dist


def salary_25():
    '''
    This function calculates number of people having salary above 25%

    Returns
    -------
    xmean : TYPE
        mean value.
    cdist : TYPE
        cumalative distribution function.
    indx : TYPE
        index of cumulative distribution function.
    xlow : TYPE
        random variable value at 0.75.

    '''
    # Calculate the mean value.
    xmean = np.sum(x_dist*y_dist)

    # Calculate the cumulative frequency.
    cdist = np.cumsum(y_dist)

    # Find the value of x which will give number of people having salary above 25%.
    indx = np.argmin(np.abs(cdist - 0.75))
    xlow = oedge[indx]

    return xmean, cdist, indx, xlow


def prob_dist_plot():
    '''
    this function returns histogram with people having salary above 25% and the
    average salary.

    Parameters
    ----------
    hist : Histrogram


    Returns
    -------
    None.

    '''
    # Plot the probability distribution.
    plt.figure(0, dpi=300)
    plt.bar(x_dist, y_dist, width=0.9*w_dist, color="green")
    plt.xlabel("Salary", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    # Plot mean value.
    text = ''' Mean Value W: {} '''.format(xmean.astype(int))
    plt.plot([xmean, xmean], [0.0, max(y_dist)], c="red", label=text)
    plt.bar(x_dist[indx:], y_dist[indx:],
            width=0.9*w_dist[indx:], color='orange')
    # Plot the Salary above 25% in orange.
    text = ''' 25% people have \n salary above X: {} '''.format(
        xlow.astype(int))
    plt.plot([xlow, xlow], [0.0, max(y_dist)], c='yellow', label=text)
    plt.title("Salaries in Europe")
    plt.legend()
    plt.savefig("Salary.png", dpi=300)
    plt.show()


# Main Function
salary_file, ohist, oedge = file_name()
x_dist, w_dist, y_dist = bin_cent()
xmean, cdist, indx, xlow = salary_25()
prob_dist_plot()



