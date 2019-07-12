import quality
import global_variables
import genetic
import random
import copy


from deap import base
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

print_all = 1 # global variable to use print 
#n_individuals = 100  

#we create the attribute fitness, This is general for all individuals
##creator.create("FitnessMax", base.Fitness, weights=(1.0,))

#we create the individual
##creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox() # global toolbox
##toolbox.register("attr_bool", quality.init_geometric, global_variables.num_drones)
toolbox.register("attr_bool", quality.init_geometric, global_variables.num_drones)

# initilize random
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_bool)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

## init guessed
toolbox.register("guessed_individ", quality.initPredefIndiv, creator.Individual)
toolbox.register("guessed_population", quality.initPredefPop, list, toolbox.guessed_individ)

toolbox.register("evaluate_coverage", quality.evaluate_coverage)
toolbox.register("evaluate_tolerance", quality.evaluate_tolerance)
toolbox.register("evaluate_redundancy", quality.evaluate_redundancy)
toolbox.register("evaluate_weighted", quality.evaluate_weighted)

# crossover
toolbox.register("mate", tools.cxTwoPoint)
# mutation
toolbox.register("mutate", genetic.shift_mutation)
# selection
toolbox.register("select", tools.selTournament, tournsize=3)
#toolbox.register("select", tools.selRoulette) 
# elitism
toolbox.register("select2", tools.selBest) 

class Population(object):
	""" It generates a population objetc with the following attributes:
		   pop: list of individuals
		   pxover = crossover probability
		   pmut = mutation probability
		   per_xover = percentage of the offspring created by crossover
		   per_mut = percentage of the offspring created by mutation
		   per_elit = percentage of the offspring created by elitism
		   n_individuals = number of individuals forming  the population
		   id_population = identification of the population
		   function = function used to evaluated the individuals (fitness function)
		   best_current_fitness = best fitness obtained so far in the evolution
		   covergence_generation = generation that obtains the best solution so far in the evolution
		   xover_offspring = list of offspring individuals for crossover operation
		   mut_offspring = lisf of offspring individuals for mutation operation
		   elit_offspring = list of offspring individuals for elitism
		   fits = list of fitness of the current population
		   best_list = list that stores the best individual of each iteration
		   toolbox = toolbox of deap
	"""
	def __init__(self, pop, pxover, pmut, per_xover, per_mut, per_elit, n_individuals, id_population, function, toolbox):
		self.pop = pop 
		self.pxover = pxover 
		self.pmut = pmut
		self.per_xover = per_xover
		self.per_mut = per_mut
		self.per_elit = per_elit
		self.n_individuals = n_individuals
		self.id = id_population
		self.function = function
		self.toolbox = toolbox
		self.best_current_fitness = None
		self.convergence_generation = None
		self.xover_offspring = list()
		self.mut_offspring = list()
		self.elit_offspring = list()
		self.offspring = list()
		self.fits = list()
		self.best_list = list()
		
	def initial_evaluation(self):
		""" This method executes the initial evaluation of the invidual of the population"""
		fitnesses = list(map(self.function, self.pop)) # create list of fitness
		for ind, fit in zip(self.pop, fitnesses):
			ind.fitness.values = fit # add the fitness to the individual
			##print(" fit ", fit)
		if print_all == 1:
			print("Individuos evaluados %d" % len(self.pop), end ="")
			
	def crossover(self):
		""" This method executes the crossover operation"""
		self.xover_offspring = self.toolbox.select(self.pop, int(self.n_individuals*self.per_xover)) # crossover selection
		self.xover_offspring = list(map(self.toolbox.clone, self.xover_offspring))
		selected_individuals = len(self.xover_offspring)
		if print_all == 1:
			print("Inds for crossover: ", selected_individuals, end ="")
		if self.n_individuals % 2 != 0:
			print("\tNumber of inds to cross must be even", end ="")
			print("\tCurrent number of inds to cross is %d" % self.n_individuals),
		for child1, child2 in zip(self.xover_offspring[::2], self.xover_offspring[1::2]):
			if random.random() < self.pxover: # crossover probability
				self.toolbox.mate(child1, child2)
				#toolbox.mate2(child1, child2)
				del child1.fitness.values # removing fitness
				del child2.fitness.values
				
	def mutation(self):
		""" This method executes the mutation operation"""
		self.mut_offspring = self.toolbox.select(self.pop, int(self.n_individuals*self.per_mut))
		selected_individuals = len(self.mut_offspring)
		if print_all == 1:
			print("\tInds for mutation: %d" % selected_individuals, end ="")
		for mutant in self.mut_offspring: # mutation probability
			if random.random() < self.pmut:
				#toolbox.mutate(mutant)
				self.toolbox.mutate(mutant,0.1)
				del mutant.fitness.values
	
	def elitism(self):
		""" This method applies elitism"""
		self.elit_offspring = self.toolbox.select2(self.pop, int(self.n_individuals* self.per_elit))
		selected_individuals = len(self.elit_offspring)
		if print_all == 1:
			print("\tInds for elitism: %d" % selected_individuals, end ="")
	
	def select_migration_elitism(self, rate):
		""" This method select the invidividual that migrate using elitism. It receives as input the % rate"""
		mig = tools.selBest(self.pop, int(self.n_individuals * rate))
		mig2 = copy.deepcopy(mig)
		return mig2
	
	def select_migration_random(self, rate):
		""" This method select the invidividual that migrate. It receives as input the % rate"""
		mig = list()
		for i in range(int(self.n_individuals * rate)):
			index = random.randint(0,len(self.pop)-1)
			mig.append(copy.deepcopy(self.pop[index]))
		return mig
	
	def insert_migration(self, migrants):
		""" This method inserts the new individual in the current population"""
		# IMPORTANT CHECK, when different fitness functions are used, the individuals
		# have to be evaluated under the new fitness function
		self.pop = self.pop + migrants
			
	def update_population(self):
		""" This methods update the current population
			1) evaluate the new offspring
			2) update population
			3) update the list of fitness
		"""
		self.offspring = self.xover_offspring + self.mut_offspring + self.elit_offspring		
		invalid_ind = [ind for ind in self.offspring if not ind.fitness.valid]
		fitnesses = map(self.function, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit
		
		if print_all == 1:
			##print("\tEvaluados %d individuos" % len(invalid_ind), end ="")
			print("\tEvaluados %d individuos" % len(invalid_ind))
			
		# update the population
		self.pop[:] = self.offspring
		self.fits = [ind.fitness.values[0] for ind in self.pop]
	
	def update_best(self, generation):
		""" this method updates: 
		the list of best individuals throughout the evolution
		the best individual found so far in the evolution
		the convergence generation
		"""
		ind = tools.selBest(self.pop, 1)[0]
		##if ind.fitness.values is None :
			##print("============================ A Value is Null ==========================================")
		self.best_list.append(ind.fitness.values)
		##print("\tself.best_current_fitness ", self.best_current_fitness, " while ind.fitness.values ", ind.fitness.values)
		if self.best_current_fitness is None or ind.fitness.values > self.best_current_fitness:
			self.best_current_fitness = ind.fitness.values
			self.convergence_generation = generation

	def statistics(self):
		""" This method calculates the statistics over the current list of fitness"""
		length = len(self.pop)
		mean = sum(self.fits) / length
		sum2 = sum(x*x for x in self.fits)
		std = abs(sum2 / length - mean**2)**0.5
		
		##print("****STATISTICS****")
		##print("Min %s" % min(self.fits), end ="")
		##print("\tMax %s" % max(self.fits), end ="")
		##print("\tAvg %s" % mean, end ="")
		##print("\tStd %s" % std)
		print("min, max, mean, std :", min(self.fits), max(self.fits), mean, std)
		return [ min(self.fits), max(self.fits), mean, std ]
	
	def plot_best(self):
		plt.figure()
		plt.plot(self.best_list)
		plt.xlabel("Generations")
		plt.ylabel("Fitness")
		name = str(self.id_population)+".png"
		plt.savefig(name)		

class Results (object):
	""" the objects of this class store the results of the evolution"""
	def __init__(self,best, best_fitness, best_evolution, identification, generation):
		self.best = best
		self.best_fitness = best_fitness
		self.best_evolution = best_evolution
		self.id = identification
		self.convergence_generation = generation

def migration_ring(population_list, migration_rate):
	""" This function implements the ring migration scheme as described in 
	http://www.pohlheim.com/Papers/mpga_gal95/gal2_5.html#Multiple Populations
	The individuals are transferred between directionally adjacent subpopulations
	"""
	mig_list = list()
	for p in population_list:
		mig_list.append(p.select_migration_elitism(migration_rate))
		#mig_list.append(p.select_migration_random(migration_rate))			
	# ring migration scheme
	## population_list[0:-1] <=> population_list-last element. As population_list[0:-3] <=> population_list- 3 last elements
	## enumerate(population_list[0:-1]): list (indexele,ele)
	for i, j in enumerate(population_list[0:-1]):
		j.pop = j.pop + mig_list[i+1]
		
	population_list[-1].pop = population_list[-1].pop + mig_list[0]
				
def migration_mesh(population_list, migration_rate, size):
	""" This fucntion implements the mesh migration ring as described in
	http://www.pohlheim.com/Papers/mpga_gal95/gal2_5.html#Multiple Populations
	The individuals may migrate from any subpopulation to another.
	"""
	mig_list = list()
	for p in population_list:
		mig_list= mig_list + p.select_migration_elitism(migration_rate) # all migrant together
		#mig_list+=p.select_migration_random(migration_rate)
	
	for p in population_list:
		m_aux_list = list()
		for m in range(0, int(migration_rate*size)-1):
			m_aux_list.append(mig_list[random.randint(0,len(mig_list)-1)])
		p.pop = p.pop + m_aux_list

def migration_all_to_one(population_list, migration_rate):
	""" this function implements all_to_one migration, the best individuals of each isolated population
	go to the last population, which implement the weighting fitness function"""
	
	for i in range(0,len(population_list)-2): # the last population is always the one tha applies the weighting function
		population_list[-1].pop = population_list[-1].pop + population_list[i].select_migration_elitism(migration_rate)
		
def ga_multi_population(argument,type_algo,i,k):
	""" this function implement the multi-objective genetic algorithm"""
	random.seed() # seed for random numbers
	
	n_individuals = 60
	##NGEN = 150 # number of generations is the same for all the populations   
	NGEN = 150
	# we create the random lisf of inividual for each population
	if argument is None :
		pop1 = toolbox.population(n_individuals) # focused on coverage 
		pop2 = toolbox.population(n_individuals) # focused on fault tolerance
		pop3 = toolbox.population(n_individuals) # focused on connections
		pop4 = toolbox.population(n_individuals) # weighted 
	else :
		n_individuals = len(argument)-1## given that first argument is the executable
		pop1 = toolbox.guessed_population(argument) # focused on coverage 
		pop2 = toolbox.guessed_population(argument) # focused on fault tolerance
		pop3 = toolbox.guessed_population(argument) # focused on connections
		pop4 = toolbox.guessed_population(argument) # weighted 

	# layout of the offspring for each population
	#per_xover = 0.5
	#per_mut = 0.4
	#per_elit = 0.1
	
	per_xover1 = 0.8
	per_mut1 = 0.1
	per_elit1 = 0.1
	
	per_xover2 = 0.7
	per_mut2 = 0.2
	per_elit2 = 0.1

	per_xover3 = 0.6
	per_mut3 = 0.3
	per_elit3 = 0.1

	per_xover4 = 0.5
	per_mut4 = 0.4
	per_elit4 = 0.1
	
	# probabilities of crossover and mutation
	p_xover= 0.6
	p_mut = 0.05
	
	#p_xover1= 0.6
	#p_mut1 = 0.05
	
	#p_xover2= 0.5
	#p_mut2 = 0.1
	
	#p_xover3= 0.4
	#p_mut3 = 0.15
	
	#p_xover4= 0.5
	#p_mut4 = 0.2
	
	#function1 = toolbox.evaluate_coverage
	#function2 = toolbox.evaluate_tolerance
	#function3 = toolbox.evaluate_redundancy
	#function1 = toolbox.evaluate_weighted
	#function2 = toolbox.evaluate_weighted
	#function3 = toolbox.evaluate_weighted
	function4 = toolbox.evaluate_weighted
	
	# IMPORTANT CHECK per_xover + per_mut + per_elet = 1
	#pop, pxover, pmut, per_xover, per_mut, per_elit, n_individuals, id
	Pop1 = Population(pop1, p_xover, p_mut, per_xover1, per_mut1, per_elit1, n_individuals, 1, function4, toolbox)
	Pop2 = Population(pop2, p_xover, p_mut, per_xover2, per_mut2, per_elit2, n_individuals, 2, function4, toolbox)
	Pop3 = Population(pop3, p_xover, p_mut, per_xover3, per_mut3, per_elit3, n_individuals, 3, function4, toolbox)
	Pop4 = Population(pop4, p_xover, p_mut, per_xover4, per_mut4, per_elit4, n_individuals, 4, function4, toolbox)

	migration_rate = 0.1
	
	population_list = list() # this list stores all the populations
	population_list.append(Pop1)
	population_list.append(Pop2)
	population_list.append(Pop3)
	population_list.append(Pop4)
	
	migration = True
	print("START EVALUATION")
	
	# initial evaluations
	for p in population_list:
		p.initial_evaluation()
		##p.statistics()
	print("END COMPUTATIONS OF INITIAL EVALUATION")
	
	# evolving the generations
	##
	notable_fits=[[] for i in range(NGEN)]
	##
	for g in range(NGEN):
		print("-- GENERATION %d --" % g)
		
		#if g%5 == 0 and migration == True:
		if g%global_variables.m_rate == 0 and migration == True:
			migration_ring(population_list, migration_rate)
			#migration_mesh(population_list, migration_rate, n_individuals)
			#migration_all_to_one(population_list, migration_rate)
		else:
			for p in population_list:
				p.crossover()
				p.mutation()
				p.elitism()
				p.update_population()
				p.update_best(g)
				if g > 0 :
					notable_fits[g]=p.statistics()

			## suppose here is where to evaluate each generations : best/worst results
	prefix="res_stats/multi_population/"
	if argument is None :
		prefix=prefix+"usual"+str(k)+"-"
	else :
		prefix=prefix+"preprocess"+str(k)+"-"

	#[print("here data in notable fits ", ele) for ele in notable_fits ]

	##for g in range(NGEN):
	f=open(prefix+str(i),"w")
	for k in range(1,len(notable_fits)) :
		strtowrite=""
		for j in range(len(notable_fits[k])) :
			if j == len(notable_fits[k])-1 :
				strtowrite=strtowrite+str(notable_fits[k][j])
			else :
				strtowrite=strtowrite+str(notable_fits[k][j])+","
		f.write(strtowrite)
		f.write("\n")
		## Note that there is a skip in the notable fits records that are due to the iteration for the migration
		## print(" value is ", strtowrite)
	f.close()

	print("-- END OF EVOLUTION --")
	
	result_list = list()
	for p in population_list:
		best_sol = tools.selBest(p.pop, 1)[0] # best individual
		##print("New best solution ", best_sol)
		#print("convergence gen ", p.convergence_generation)
		best_fit = best_sol.fitness.values[0] # best fitness
		best_list = p.best_list # best list of fitnesses
		result_list.append(Results(best_sol, best_fit,best_list,p.id, p.convergence_generation))

	##[print("content result_list : ", p.best_fitness, p.best_evolution, p.id, p.convergence_generation) for p in result_list]
	return result_list

