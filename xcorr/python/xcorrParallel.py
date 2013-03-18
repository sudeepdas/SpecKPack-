import multiprocessing as mp
import xcorrReal, xcorrFourier

class worker(mp.Process):
    """
    @brief class that dequeues a task from the input Queue, 
    does the task work on a new Process, and then stores the  
    the result on a output result Queue
    """
    
    def __init__(self, task_queue, result_queue):
        
        mp.Process.__init__(self) # initialize a new Process for each worker
        self.task_queue   = task_queue # the Queue of tasks to do 
        self.result_queue = result_queue # the Queue to store the results

    def run(self):
        """
        @brief start the worker class doing the tasks until there
        are none left
        """
        
        # pull tasks until there are none left
        while True:
            # dequeue the next task
            next_task = self.task_queue.get()
            
            # task=None means this worker is finished
            if next_task is None:
                # make sure we tell the queue we finished the task
                self.task_queue.task_done()
                break
                
            # do the work by calling the task    
            answer = next_task()
            
            # store the answer
            self.result_queue.put(answer)
            
            # make sure we tell the queue we finished the task
            self.task_queue.task_done()
        
        return 0
    
class realCorrTask(object):
    """
    @brief a class representing a 'task' where the real space correlation between
    a data map and random map is computed
    """
    def __init__(self, params, i, paramsToUpdate):
        self.params         = params # the relevant parameters
        self.num            = i # the random map number to use
        self.paramsToUpdate = paramsToUpdate # dictionary of parameters to update
        
    def __call__(self):
        
        # update the parameters
        self.updateParams()
        
        # compute the real space correlation, given the parameters
        err = xcorrReal.realSpaceCorr(self.params)
        if err: exit()
            
        return 0

    def updateParams(self):
        """
        @brief update the parameters associated with this task class
        """
        # loop over each parameter name and new value in the dictionary
        for key, val in self.paramsToUpdate.items():
            # update each element of a list
            if type(val) == list:
                for i in range(len(val)):
                    self.params[key][i] = val[i] %self.num
            else:
                self.params[key] = val %self.num
        
        return 0

