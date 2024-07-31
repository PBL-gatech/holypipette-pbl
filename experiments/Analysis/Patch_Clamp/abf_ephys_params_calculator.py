# calculating ephys params from abfs
import numpy as np
import pyabf
import matplotlib.pyplot as plt

# convert abf to class
class data:
    def __init__(self,time,response,command,sampleRate):
        self.time = time
        self.response = response
        self.command = command
        self.sampleRate = sampleRate
        self.numSweeps = 1
        
def abf2class(abf):
    for sweepNumber in abf.sweepList:
        abf.setSweep(sweepNumber)
        if sweepNumber == 0:
            myData = data(time=abf.sweepX,response=abf.sweepY,command=abf.sweepC,sampleRate=int(1/(abf.sweepX[1]-abf.sweepX[0])))
            # print the length of the time, resposne, command arrays. print the sample rate,  and the number of sweeps.
            print("Time: ",len(myData.time))
            print("Response: ",len(myData.response))
            print("Command: ",len(myData.command))
            print("Sample Rate: ",myData.sampleRate)
            print("Number of Sweeps: ",myData.numSweeps)
        else:
            myData.response = np.vstack((myData.response,abf.sweepY))
            myData.command = np.vstack((myData.command,abf.sweepC))
            myData.numSweeps = myData.numSweeps + 1
        
    return myData


def raw_data_plot(myData):
#plot the command and response curves together in a 2 by 1 plot
    plt.figure(figsize=(8, 5))
    plt.subplot(2,1,1)
    plt.plot(myData.time,myData.command.T)
    plt.title('Command')
    plt.subplot(2,1,2)
    plt.plot(myData.time,myData.response.T)
    plt.title('Response')
    plt.show()
    

def getResponseDataSweep(d,sweepNum):
    return d.current[sweepNum,:]

def getCommandSweep(d,sweepNum):
    return d.command[sweepNum,:]

def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b

def calc_pas_params(d,filename,base_fn): # filename is the image path, base_fn is the name of the abf
    # initialize the array to save all the parameters
    n_sweeps = d.numSweeps
    print ("Calculating passive properties for ",n_sweeps," sweeps")
    n_params = 5 + 1 # for fn
    all_data = np.empty((n_sweeps, n_params))
    try:
        # voltage_data = getResponseDataSweep(d,0)
        voltage_data = d.current[0,:]
        print("ready.")
    except:
        print("Error: Could not get response data for sweep 0")
        return
    # for each sweep in the abf, find the passive properties and save to the array 
    for sweep in [1]: 
        voltage_data = getResponseDataSweep(d,sweep)
        dt = 1/d.sampleRate
        command_current = getCommandSweep(d,sweep)
        del_com = np.diff(command_current)
        starts = np.where(del_com<0)
        ends = np.where(del_com>0)
        # these should be for passive properties (ie 1st step down)
        const = 0
        passive_start = starts[0][0] + const
        passive_end = ends[0][0] - const

        mean1 = np.mean(voltage_data[0 : passive_start-1])  #calculate Rm/input_resistance
        mean2 = np.mean(voltage_data[int(passive_start + (0.1 / dt)) : passive_end])

        holding = np.mean(command_current[0: passive_start-10])
        pas_stim = np.mean(command_current[passive_start + 10 : passive_start + 110]) - holding

        input_resistance = (abs(mean1-mean2) / abs(pas_stim) ) * 1000 # Steady state delta V/delta I
        
        resting = mean1 - (input_resistance * holding) / 1000

        #plot response and command curves
        plt.figure(figsize=(8, 5))
        plt.plot(d.time, voltage_data)
        plt.plot(d.time, command_current)

        import scipy.optimize

        X1 = d.time[passive_start : int((passive_start + (0.1 / dt)))]           #calculate membrane tau
        Y1 = voltage_data[passive_start : int((passive_start + (0.1 / dt)))]

        # p0 = (100, 17, 1000)
        p0 = (6.123e12, 54.097, -48.3)
        try:
            params, cv = scipy.optimize.curve_fit(monoExp, X1[::50], Y1[::50], p0, maxfev = 100000000)
            m, t, b = params
            print("samplerate: ",sampleRate)
            sampleRate = int(1 / dt / 1000)+1
            membrane_tau =  ((1 / t) / sampleRate) * 1e6 / abs(pas_stim)
            membrane_capacitance = membrane_tau / input_resistance *1000
            print("tau: ",membrane_tau)
            print("cap: ",membrane_capacitance)
            print("step down: ",pas_stim)
        except:
            m = 0
            t = 0
            b = 0
            membrane_tau = 0
            membrane_capacitance = 0
        # find error in fit
        fit_err = np.average(abs((monoExp(d.time[passive_start:passive_end],m,t,b)-voltage_data[passive_start:passive_end])))

        if 1:
            # find limits for the y-axis
            max_lim = np.max(voltage_data[passive_start:passive_end]) + 5
            min_lim = np.min(voltage_data[passive_start:passive_end]) - 5
            
            # plot for checking fitting
            plt.plot(d.time,voltage_data)
            plt.plot(X1,Y1)
            plt.plot(d.time,monoExp(d.time, m, t, b))
            plt.scatter([d.time[passive_start],d.time[passive_end]],[voltage_data[passive_start],voltage_data[passive_end]],c='red')
            plt.ylim([min_lim,max_lim])
            plt.xlim([.4,1.1])
            plt.savefig(filename+".png")
            plt.show()
            plt.clf()

        print("Tau: ",membrane_tau)
        print("Capacitance: ",membrane_capacitance)

        # delete this
        base_fn = 111

        all_data[sweep,:] = [int(base_fn),membrane_tau, input_resistance, membrane_capacitance, resting, fit_err]
    return all_data 



abf_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Analysis\Patch_Clamp\24418016.abf"
abf  = pyabf.ABF(abf_path)
myData = abf2class(abf)
raw_data_plot(myData)
all_data = calc_pas_params(myData,abf_path,abf.abfID)
print(all_data)
