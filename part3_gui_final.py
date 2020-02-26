import sys
from PyQt5.QtWidgets import *
from •	Delivering efficient, prompt and friendly service to customers
•	Training coworkers with coffee making techniques and general café responsibilities
•	Keeping work area clean and tidy 
•	Cash management by handling the register.QtCore import *
from PyQt5.QtGui import *

import numpy as np
import math
import random
import time
import sys
from random import shuffle
import mysql.connector
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.image as mpimg



############### SOLVER ##################
def create_matrix(number_of_cities):
    '''
    create n*n matrix

    '''
    number_of_cities = int(number_of_cities)
    matrix = np.zeros(shape=(number_of_cities, number_of_cities))
    return matrix


def distances_between_cities(matrix, points_int):
    '''
    find distances between city and fill the matrix

    '''
    i = 0
    j = 0
    for x in points_int:
        for y in points_int:
            matrix[i][j] = math.sqrt((y[0]-x[0])**2 + (y[1]-x[1])**2)
            j += 1
        i += 1
        j = 0


def calculate_distance(route, points):
    '''
    calculate distace of given route

    '''
    distance = 0
    i = 1
    for x in route:
        for y in route[i::]:
            a = points[y][0]-points[x][0]
            b = points[y][1]-points[x][1]

            distance += math.sqrt(a**2+b**2)
            i += 1
            break
    x = points[route[-1]][0] - points[route[0]][0]
    y = points[route[-1]][1] - points[route[0]][1]
    distance += math.sqrt(x**2+b**2)

    return distance


def greedy_solver(matrix):
    '''
    Computes the initial solution (nearest neighbour strategy)

    '''
    node = random.randrange(len(matrix))
    result = [node]

    nodes_to_visit = list(range(len(matrix)))
    nodes_to_visit.remove(node)

    while nodes_to_visit:
        nearest_node = min([(matrix[node][j], j)
                            for j in nodes_to_visit], key=lambda x: x[0])
        node = nearest_node[1]
        nodes_to_visit.remove(node)
        result.append(node)

    return result


def acceptance_probability(candidate, current, temp):
    return math.exp(-abs(candidate - current)/temp)


def k_opt(init_route, init_distance, Dimension, k, points_int):
    for i in range(Dimension - k):
        lst = init_route[i:i+k]
        route1_dist = calculate_distance(lst, points_int)
        route2 = random.sample(lst, k)
        route2_dist = calculate_distance(route2, points_int)

        if route1_dist > route2_dist:
            temp_route = init_route
            temp_route[i:i+k] = route2
            temp = calculate_distance(temp_route, points_int)

            if temp < init_distance:
                init_route = temp_route
                init_distance = calculate_distance(init_route, points_int)


try:
    mydb = mysql.connector.connect(
        host="mysql.ict.griffith.edu.au",
        user="s5124191",
        passwd="uiJN7cTb"
    )
except:
    sys.exit("Cannot connect to the database ")

mycursor = mydb.cursor(buffered=True)
mycursor.autocommit = False


mycursor.execute("USE 1810ICTdb;")
mydb.commit()

mycursor.execute("""CREATE TABLE IF NOT EXISTS Problem (
                    Name varchar(32) NOT NULL,
                    Size int(11) NOT NULL,
                    Comment varchar(255) DEFAULT NULL,
                    CONSTRAINT PPK PRIMARY KEY (Name)
                    );""")
mydb.commit()

mycursor.execute("""CREATE TABLE IF NOT EXISTS Cities (
                    Name varchar(32) NOT NULL,
                    ID int(11) NOT NULL,
                    x double NOT NULL,
                    y double NOT NULL,
                    CONSTRAINT CPK PRIMARY KEY (Name, ID),
                    CONSTRAINT PName FOREIGN KEY (Name) REFERENCES Problem (Name) ON DELETE CASCADE
                    );""")
mydb.commit()

mycursor.execute("""CREATE TABLE IF NOT EXISTS Solution (
                    SolutionID int(11) NOT NULL AUTO_INCREMENT,
                    ProblemName varchar(32) NOT NULL,
                    TourLength double NOT NULL,
                    Date date DEFAULT NULL,
                    Author varchar(32) DEFAULT NULL,
                    Algorithm varchar(32) DEFAULT NULL,
                    RunningTime int(11) DEFAULT NULL,
                    Tour mediumtext NOT NULL,
                    CONSTRAINT SPK PRIMARY KEY (SolutionID),
                    CONSTRAINT SolPName FOREIGN KEY (ProblemName) REFERENCES Problem (Name) ON DELETE CASCADE
                    );""")
mydb.commit()



############# GUI ###############
class Window(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TSP")
        
        layout = QVBoxLayout()
        
        ################ Tabs ###################

        self.tabs = QTabWidget()

        self.addTspTab = QWidget()
        self.solveTab = QWidget()
        self.solutionsTab = QWidget()

        self.tabs.addTab(self.addTspTab, 'ADD')
        self.tabs.addTab(self.solveTab, 'SOLVE')
        self.tabs.addTab(self.solutionsTab, 'SOLUTIONS')

        ############### Widgets #################

        layout.addWidget(self.tabs)

        addLayout = QVBoxLayout()
        solveLayout = QVBoxLayout()
        solutionLayout = QVBoxLayout()

        ################# ADD tsp problem ######################
        intro = QLabel()
        intro.setText("""How to use the program\n
        1. ADD: tpye the name of the tsp file and click add\n
        2. SOLVE: choose the problem to solve, and set the time to solve the problem. \n
        \tThen press SOLVE button to solve. Please wait until soving is finished. \n
        \tSHOW GRAPH will present the graph that is based on the solver just created.\n
        3. SOLUTIONS: choose any of the solutions and click SHOW GRAPH to see the solution.
        """)
        addLayout.addWidget(intro)

        self.tspProblemName = QLineEdit(self)
        self.tspProblemName.setPlaceholderText('eg) a280')
        addLayout.addWidget(self.tspProblemName)

        instruction2 = QLabel()
        instruction2.setText("Problems which are already in the database.")
        addLayout.addWidget(instruction2)

        self.listOfProblems = QListWidget()

        sql = """SELECT Name FROM Problem"""
        mycursor.execute(sql)
        for i in mycursor:
            self.listOfProblems.addItem(i[0])
        self.listOfProblems.clicked.connect(self.chooseProblems)
        addLayout.addWidget(self.listOfProblems)

        addButton = QPushButton("ADD", self)
        addButton.clicked.connect(self.addProblems)
        addLayout.addWidget(addButton)


        ############## Show list of problems and solve ############
        instruction1 = QLabel()
        instruction1.setText("Saved problems")
        solveLayout.addWidget(instruction1)

        self.listOfProblems = QListWidget()

        # add problems to the list 
        sql = """SELECT Name FROM Problem"""
        mycursor.execute(sql)
        for i in mycursor:
            self.listOfProblems.addItem(i[0])
        self.listOfProblems.clicked.connect(self.chooseProblems)
        solveLayout.addWidget(self.listOfProblems)
        
        # solving time 
        instruction2 = QLabel()
        instruction2.setText("\nSet time (seconds)")
        solveLayout.addWidget(instruction2)
        self.solvingTime = QSpinBox(self)
        self.solvingTime.setMaximum(300)
        self.solvingTime.valueChanged.connect(self.setSolvingTime)
        solveLayout.addWidget(self.solvingTime)

        # solve button
        solveButton = QPushButton("SOLVE", self)
        solveButton.clicked.connect(self.solveProblems)
        solveLayout.addWidget(solveButton)

        instruction4 = QLabel()
        instruction4.setText("Please wait for a given set time\n")
        solveLayout.addWidget(instruction4)

        # Canvas 
        self.solve_figure = plt.figure()
        self.canvas = FigureCanvas(self.solve_figure)
        solveLayout.addWidget(self.canvas)

        # show graph button
        graphButton = QPushButton("SHOW GRAPH", self)
        graphButton.clicked.connect(self.solve_plot)
        solveLayout.addWidget(graphButton)

        addToSolutionButton = QPushButton("ADD TO SOLUTIONS", self)
        addToSolutionButton.clicked.connect(self.addToSolution)
        solveLayout.addWidget(addToSolutionButton)


        ############# Show list of solutions ##############
        instruction3 = QLabel()
        instruction3.setText("Saved solutions (ID, author, filename, given time, total distance)")
        solutionLayout.addWidget(instruction3)

        self.listOfSolutions = QListWidget()
        sql = """SELECT ProblemName, RunningTime, TourLength, SolutionID, Author FROM Solution ORDER BY `SolutionID` DESC"""
        mycursor.execute(sql)
        for i in mycursor:
            self.listOfSolutions.addItem(str(i[3])+ ', ' + str(i[4]) + ', ' + i[0]+ ', ' + str(i[1])+ ', ' + str(i[2]))
        self.listOfSolutions.clicked.connect(self.chooseSolutions)
        solutionLayout.addWidget(self.listOfSolutions)
        
        # Canvas
        self.solution_figure = plt.figure()
        self.solution_canvas = FigureCanvas(self.solution_figure)
        solutionLayout.addWidget(self.solution_canvas)

        # show graph button
        sol_graphButton = QPushButton("SHOW GRAPH", self)
        sol_graphButton.clicked.connect(self.solution_plot)
        solutionLayout.addWidget(sol_graphButton)

        ################# Set layouts #####################
        widget = QWidget()

        widget.setLayout(layout)
        self.addTspTab.setLayout(addLayout)
        self.solveTab.setLayout(solveLayout)
        self.solutionsTab.setLayout(solutionLayout)
        
        self.setCentralWidget(widget)

        self.tspProblemName.update()
        self.listOfProblems.update()


    def addProblems(self):
        problem = self.tspProblemName.text()
        tspfile = problem + '.tsp'
        # print(tspfile)

        try:
            with open(tspfile) as f:
                tsp = f.read()

            Comment = str(tsp.split('\n')[2].split(':')[1])
            Dimension = int(tsp.split('\n')[3].split()[-1])

            add = (problem, Dimension, Comment)
            sql = "INSERT INTO Problem (`Name`, `Size`, `Comment`) VALUES (%s, %s, %s);"
            mycursor.execute(sql, add)
            mydb.commit()

            node_lines = tsp.split('\n')[6:-2]
            points_string = [p.split()[0:] for p in node_lines]
            for p in points_string:
                add = (problem, int(p[0]), float(p[1]), float(p[2]))
                sql = "INSERT INTO Cities (`Name`, `ID`, `x`, `y`) VALUES (%s, %s, %s, %s);"
                mycursor.execute(sql, add)
                mydb.commit()
        except:
            print("Cannot process ADD ")
        
        sql = """SELECT Name FROM Problem"""
        mycursor.execute(sql)
        self.listOfProblems.clear()
        for i in mycursor:
            self.listOfProblems.addItem(i[0])
        
    def setSolvingTime(self):
        self.setTime = self.solvingTime.value()

    def addToSolution(self):
        Author = "s5124191"
        Algorithm = "Simulated Annealing"
        date = "CURDATE()"

        add_solution = (self.ProblemName, self.TourLength, self.RunningTime, self.Tour, Author, Algorithm, date)
        sql = """INSERT INTO Solution (`ProblemName`, `TourLength`, `RunningTime`, `Tour`, `Author`, `Algorithm`, `Date`) 
                VALUES (%s, %s, %s, %s, %s, %s, %s);"""
        mycursor.execute(sql, add_solution)
        mydb.commit()

            # Update list of solutions
        sql = """SELECT ProblemName, RunningTime, TourLength, SolutionID FROM Solution"""
        mycursor.execute(sql)
        self.listOfSolutions.clear()
        for i in mycursor:
            self.listOfSolutions.addItem(str(i[3]) + ', ' + i[0] + ', ' + str(i[1]) + ', ' + str(i[2]))



    def chooseProblems(self):
        self.chosenProblem = self.listOfProblems.currentItem()

    
    def chooseSolutions(self):
        self.chosenSolutions = self.listOfSolutions.currentItem()
        sol = self.chosenSolutions.text().split(', ')

        points_int = []
        sql = "SELECT * FROM `Cities` WHERE `Name` = %s ORDER BY `ID` ASC"
        mycursor.execute(sql, (sol[2],))
        mydb.commit()

        for x in mycursor:
            points_int.append([x[2], x[3]])

        sql = "SELECT Tour FROM Solution WHERE SolutionID = %s"
        mycursor.execute(sql, (sol[0],))
        mydb.commit()

        self.xs = []
        self.ys = []
        self.warningSolution = False

        try:
        
            for i in mycursor:
                route = i[0]
        
            if '[' in route:
                route = route[1:-1]

            if ', ' in route:
                route = route.split(', ')
            elif ',' in route:
                route = route.split(',')
            else:
                route = route.split()

            if(type(route[0])==str):
                route = [int(i) for i in route]
        

            # print(f"len route: {len(route)}, max route: {max(route)}")
            print("\nRoute")
            print(route)
            route_len = len(route)
            route_max = max(route)

            # when the route dose not finish where it started  
            if(route_len == route_max+2 and route[-1] > 0):
                route.append(route[0])

            # when the route finishes where it started
            elif(route_len == route_max+1):
                route[-1] = route[0]

            try:
                for x in route:
                    coord = points_int[x]
                    self.xs.append(coord[0])
                    self.ys.append(coord[1])
            except IndexError:
                print("""Error occured while plot the graph. Alterated graph is generated. It is possible that the graph is not matching to original route""")
                route.remove(max(route))
                for x in route:
                    coord = points_int[x]
                    self.xs.append(coord[0])
                    self.ys.append(coord[1])

        except:
            print("Cannot process solution, please choose another solution")
            self.warningSolution = True


    def solveProblems(self):

        problem = self.chosenProblem.text()
        
        try:
            time_limit = self.setTime

            points_int = []
            sql = "SELECT * FROM `Cities` WHERE `Name` = %s ORDER BY `ID` DESC"
            mycursor.execute(sql, (problem,))
            mydb.commit()

            for x in mycursor:
                points_int.append([x[2], x[3]])

            Dimension = len(points_int)
            start_time = time.time()

            temp = Dimension
            matrix = create_matrix(Dimension)
            distances_between_cities(matrix, points_int)

            init_route = greedy_solver(matrix)
            init_distance = calculate_distance(init_route, points_int)
            initial = init_distance

            while temp > 0.001 and time.time()-start_time < time_limit:

                candidate = greedy_solver(matrix)
                candidate_distance = calculate_distance(candidate, points_int)

                k_opt(candidate, candidate_distance, Dimension, 3, points_int)
                k_opt(candidate, candidate_distance, Dimension, 4, points_int)
                k_opt(candidate, candidate_distance, Dimension, 5, points_int)

                if candidate_distance < init_distance:
                    init_distance = candidate_distance
                    init_route = candidate

                elif acceptance_probability(candidate_distance, init_distance, temp) > random.uniform(0, 1):
                    candidate = greedy_solver(matrix)
                    candidate_distance = calculate_distance(candidate, points_int)

                    k_opt(candidate, candidate_distance, Dimension, 3, points_int)
                    k_opt(candidate, candidate_distance, Dimension, 4, points_int)
                    k_opt(candidate, candidate_distance, Dimension, 5, points_int)

                    if candidate_distance < init_distance:
                        init_distance = candidate_distance
                        init_route = candidate

                temp *= 0.9995

            init_route.append(-1)
            print(f"\n\nFile Name: {problem}")
            print(f"Processing Time: {round(time.time()-start_time,2)}")
            print(f"Initial Distance using Greedy solver:\t\t\t{round(initial,3)}")
            print(
                f"Optimal Distance using Simulated Annealing with K-opt:\t{round(init_distance,3)}")
            print(
                f"Optimization: {round((initial-init_distance)*100/initial,3)}% shorter than Initial Distance")
            print("\nOptimal Route: ")
            print(init_route)

            tour_length = round(init_distance, 3)
            processing_time = round(time.time()-start_time, 2)

            init_route[-1] = init_route[0]
            self.ProblemName = problem
            self.TourLength = tour_length
            self.RunningTime = processing_time
            self.Tour = str(init_route)[1:-1]

            # Xs and Ys for the graph 
            self.xs = []; self.ys = []
            for i in init_route:
                coord = points_int[int(i)]
                self.xs.append(coord[0])
                self.ys.append(coord[1])

        except:
            print("Cannot process SOLVE")


    def solve_plot(self):
        ax = self.solve_figure.add_subplot(111)
        ax.plot(self.xs, self.ys)
        ax.set_title('Optimal Route')
        self.canvas.draw()
        self.canvas.show()
        ax.clear()
        self.hide()
        self.show()
    

    def solution_plot(self):
        if(self.warningSolution):
            ax2 = self.solution_figure.add_subplot(111)
            img = mpimg.imread('https://i.redd.it/q8upiiorecx11.png')
            ax2 = plt.imshow(img)
            self.solution_canvas.show()
            self.solution_canvas.draw()
            self.hide()
            self.show()

        else:
            ax2 = self.solution_figure.add_subplot(111)
            ax2.clear()
            ax2.plot(self.xs, self.ys)
            ax2.set_title('Optimal Route')
            self.solution_canvas.show()
            self.solution_canvas.draw()
            ax2.clear()
            self.hide()
            self.show()

    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
