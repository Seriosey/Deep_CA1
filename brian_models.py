from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from brian2 import NeuronGroup, Synapses, SpikeMonitor, StateMonitor
from brian2 import ms, mV
from brian2 import defaultclock, run
import csv
import pandas as pd
from scipy.special import i0 as bessel


# import warnings
# warnings.filterwarnings("ignore")
# BrianLogger.initialize()
# BrianLogger.suppress_hierarchy('brian2', filter_log_file=False)
# logging.console_log_level = 'FALSE'
# logging.file_log_level = 'FALSE'
# logging.file_log = False


THETA_FREQ = 8
V_AN = 20
duration = 5 * second
tfinal = 5000 * ms
defaultclock.dt = 0.02 * ms



def r2kappa(R):
    """
    recalulate kappa from R for von Misses function
    """
    if R < 0.53:
        kappa = 2 * R + R**3 + 5/6 * R**5

    elif R >= 0.53 and R < 0.85:
        kappa = -0.4 + 1.39 * R + 0.43 / (1 - R)

    elif R >= 0.85:
        kappa = 1 / (3*R - 4*R**2 + R**3)
        
    I0 = bessel(kappa)

    return kappa, I0


neuron_types = pd.read_csv('parameters/DG_CA2_Sub_CA3_CA1_EC_neuron_parameters06-30-2024_10_52_20.csv', delimiter=',')
#pd.set_option('max_columns', None)
#print(neurons.head())

eqs = """dv/dt = ((k*(v - vr)*(v - vt) - u_p + I + I_noise + 10*I_syn)/Cm)/ms : 1
         du_p/dt = (a*(b*(v-vr) - u_p))/ms  : 1
         I : 1
         I_noise : 1
         I_syn : 1
         a : 1
         b : 1
         d : 1
         Cm : 1
         k : 1
         vt : 1
         vr : 1
         Vpeak : 1
         Vmin : 1
         refr : second
       """

Types = []
NeuronTypes = {}
for index, neuron_param in neuron_types.iterrows():
    key = neuron_param['Neuron Type']
    Types.append(key)
    if key == 'CA1 Axo-axonic': key = 'CA1 Axo-Axonic'
    if key == 'CA1 Horizontal Axo-axonic': key = 'CA1 Horizontal Axo-Axonic'
    #if key == 'CA1 Oriens-Alveus': key = 'CA1 Oriens/Alveus'
    if 'CA1' in key:
        #neuron_param = neuron_types[neuron_types['Neuron Type'] == 'CA1 Pyramidal']
        
        G = NeuronGroup(1, eqs, threshold="v>=Vpeak", reset="v=Vmin; u_p+=d", refractory='refr', method="euler") #neuron['Population Size']
        G.Vpeak = neuron_param['Izh Vpeak']
        G.Vmin = neuron_param['Izh Vmin']
        G.refr = neuron_param['Refractory Period']*ms
        G.a = neuron_param['Izh a']
        G.b = neuron_param['Izh b']
        G.Cm = 1 * neuron_param['Izh C']
        G.d = neuron_param['Izh d']
        G.vr = neuron_param['Izh Vr']
        G.vt = neuron_param['Izh Vt']
        G.k = neuron_param['Izh k']
        G.I = 50
        G.run_regularly("I_noise = 100*randn()", dt=1 * ms)
        NeuronTypes[key] = G
        #print(f'Model of type {key} is created...')


print('Successfuly created neuron models for all types')

# C = NeuronTypes['CA1 Pyramidal']
# print(C.a, C.b, C.d, C.Cm, C.vr, C.vt, C.k, C.u_p)

# M = StateMonitor(C, 'v', record=True)

# run(tfinal)
# plot(M.t/ms, M.v[0])
# plt.show()

ca1params = {
            "name": "ca1pyr",
            "R": 0.2,
            "freq": THETA_FREQ,
            "mean_spike_rate": 0.5,#*20, #######
            "phase": 3.14,
        }

ec3params = {
            "name": "ec3",
            "R": 0.3,
            "freq": THETA_FREQ,
            "mean_spike_rate": 1.5,
            "phase": -1.57,

            "sigma_sp": 5.0,  # cm
            "v_an": V_AN,  # cm/sec
            "maxFiring": 8.0,  # spike sec in the center of field
            "sp_centers": -5.0,  # cm
        }

ca3params = {
            "name": "ca3pyr",
            "R": 0.3,
            "freq": THETA_FREQ,
            "mean_spike_rate": 0.5,#*20,##########
            "phase": 1.58,

            "sigma_sp": 8.0,  # cm
            "v_an": V_AN,  # cm/sec
            "maxFiring": 8.0,  # spike sec in the center of field
            "sp_centers": 5.0}


ca3_center = ca3params['sp_centers']/ V_AN + duration/second * 1000 / 2
ca3params['t_center'] = ca3_center
ca3params['sigma_sp'] *= V_AN


ec3_center = ec3params['sp_centers']/ V_AN + duration/second * 1000 / 2
ec3params['t_center'] = ec3_center
ec3params['sigma_sp'] *= V_AN

rates_template_mec = '{fr} * Hz * exp({kappa} * cos(2*pi*{freq}*Hz*t - {phase}) )'
rates_template_ca3 = '{fr} * Hz * exp({kappa} * cos(2*pi*{freq}*Hz*t - {phase}) ) * (1+ {maxFiring} * exp(-0.5 * ((t/ms - {t_center}) / {sigma_sp}) ** 2))'
rates_template_lec = '{fr} * Hz * exp({kappa} * cos(2*pi*{freq}*Hz*t - {phase}) ) * (1+ {maxFiring} * exp(-0.5 * ((t/ms - {t_center}) / {sigma_sp}) ** 2))'

#RNG = np.random.default_rng()

kappa, IO = r2kappa(ca1params["R"])
ca1params["kappa"] = kappa
ca1params["fr"] = ca1params["mean_spike_rate"] / IO

kappa, IO = r2kappa(ca3params["R"])
ca3params["kappa"] = kappa
ca3params["fr"] = ca3params["mean_spike_rate"] / IO

kappa, IO = r2kappa(ec3params["R"])
ec3params["kappa"] = kappa
ec3params["fr"] = ec3params["mean_spike_rate"] / IO

ca3rates = rates_template_ca3.format(**ca3params) # ca1
mec_rates = rates_template_mec.format(**ca1params)
lec_rates = rates_template_lec.format(**ec3params)
N = 1000 # 1000*6

ca3 = PoissonGroup(N, rates=ca3rates)
ca3_sm = SpikeMonitor(ca3)

mec = PoissonGroup(N, rates=mec_rates)
ca1_sm = SpikeMonitor(mec)

lec = PoissonGroup(N, rates=lec_rates)
ec3_sm = SpikeMonitor(lec)


print('Succesfuly created generator models')

# run(duration)

# mfr = np.asarray(ca3_sm.t/ms).size / N / (duration/second)
# firing_rate, bins = np.histogram(ca3_sm.t/ms, bins=int(duration/ms)+1, range=[0, int(duration/ms)+1])
# dbins = 0.001*(bins[1] - bins[0])
# firing_rate = firing_rate / N / dbins

# t = np.linspace(0, duration/second, 1000)
# sine = np.max(firing_rate) * 0.5 * (np.cos(2*np.pi*ca3params["freq"]*t)+1)
# fig, axes = plt.subplots(nrows=2)
# axes[0].plot(t, sine)
# axes[0].plot(0.001*bins[:-1], firing_rate)
# axes[1].scatter(ca3_sm.t/ms, ca3_sm.i, s=2)

# plt.show()


t_gen = np.arange(0.0, 10000.0, 0.1)



def get_synapses(pre, post, tau_inact, tau_rec, tau_facil, A_SE, U_SE, t_delay, conn_prob):
    """
    pre -- input stimulus
    post -- target neuron
    tau_inact -- inactivation time constant
    A_SE -- absolute synaptic strength
    U_SE -- utilization of synaptic efficacy
    tau_rec -- recovery time constant
    tau_facil -- facilitation time constant (optional)
    """

    synapses_eqs = """
    dx/dt =  z/tau_rec   : 1 (clock-driven) # recovered
    dy/dt = -y/tau_inact : 1 (clock-driven) # active
    A_SE : 1
    U_SE : 1
    tau_inact : second
    tau_rec : second
    z = 1 - x - y : 1 # inactive
    I_syn_post = A_SE*y : 1 (summed)
    """

    if tau_facil:
        synapses_eqs += """
        du/dt = -u/tau_facil : 1 (clock-driven)
        tau_facil : second
        """

        synapses_action = """
        u += U_SE*(1-u)
        y += u*x # important: update y first
        x += -u*x
        """
    else:
        synapses_action = """
        y += U_SE*x # important: update y first
        x += -U_SE*x
        """


    synapses = Synapses(pre,
                        post,
                        model=synapses_eqs,
                        on_pre=synapses_action,
                        method="exponential_euler")
    synapses.connect(p=conn_prob) #conn_prob

    # start fully recovered
    synapses.x = 1

    synapses.tau_inact = tau_inact*ms
    synapses.A_SE = A_SE*1#00
    synapses.U_SE = U_SE*1#00
    synapses.tau_rec = tau_rec*ms
    t_delay = t_delay*ms
    synapses.delay = 't_delay'


    if tau_facil:
        synapses.tau_facil = tau_facil*ms

    return synapses




synapse_types = pd.read_csv('parameters/DG_CA2_Sub_CA3_CA1_EC_conn_parameters06-30-2024_10_52_20.csv')

# synapse_type = connections[connections['Presynaptic Neuron Type'] == 'CA1 Axo-Axonic'][connections['Postsynaptic Neuron Type'] == 'CA1 Pyramidal']
# print(synapse_type)

#'''

import pickle
from pprint import pprint


neurons_list = []  # Necessary for net neurons
with (open("presimulation_files/neurons.pickle", "rb")) as openfile:
    while True:
        try:
            neurons_list = pickle.load(openfile)
        except EOFError:
            break
# pprint(neurons) # in neurons.pickle list of such dicts: {'MeanFiringRate': nan,
                                                        #    'R': 0.265625,
                                                        #    'ThetaPhase': nan,
                                                        #    'type': 'CA1 Schaffer Collateral-Associated',
                                                        #    'x_anat': -2175.0,
                                                        #    'y_anat': 1350.0,
                                                        #    'z_anat': 0},

synapses = []
with (open("presimulation_files/connections.pickle", "rb")) as openfile:
    while True:
        try:
            synapses = pickle.load(openfile)
        except EOFError:
            break

#pprint(synapses)                # {'gsyn': None,
                                # 'pconn': 0.040531372899119,
                                # 'post_idx': 3619,
                                # 'post_type': 'CA1 Trilaminar',
                                # 'pre_idx': 3619,
                                # 'pre_type': 'CA1 Trilaminar'}

generators_params = []
types = set()
neurons = []
for neuron in neurons_list:
    type = neuron['type']
    types.add(type)
    if type == 'CA1 Basket CCK+': continue
    neurons.append({'type' : type, 
                    'neuron' : NeuronTypes[type]})
    if (type == 'ca3_generator') or (type == 'lec_generator') or (type == 'mec_generator'):
        generators_params.append(neuron)
    #[NeuronTypes[type], type]

ready_synapses = []
ready_neurons = [neuron['neuron'] for neuron in neurons]

# print(types)

# net = Network(collect())

'''
for connection in synapses:

    if connection['pre_idx'] >= len(neurons) or connection['post_idx'] >= len(neurons): continue
    values = [0, 1]
    pconn = connection['pconn']
    data = random.choices(values, weights=[1 - pconn, pconn])
    if data == 1:

        # pre_neuron = neurons[connection['pre_idx']]['neuron']
        # post_neuron = neurons[connection['post_idx']]['neuron']

        pre_type = neurons[connection['pre_idx']]['type']
        post_type = neurons[connection['post_idx']]['type']

        #print(pre_type,'        ', post_type)

        # Choosing right type of synapse from synapse_types table 
        synapse_type = synapse_types[synapse_types['Presynaptic Neuron Type'] == pre_type][synapse_types['Postsynaptic Neuron Type'] == post_type]

        # Gettting parameters of synapse and connecting via function get_synapses
        if not synapse_type.empty:

            tau_d, tau_r, tau_f = synapse_type['tau_d'].iloc[0], synapse_type['tau_r'].iloc[0], synapse_type['tau_f'].iloc[0]
            A_SE, U_SE = synapse_type['g'].iloc[0], synapse_type['u'].iloc[0]
            t_delay = synapse_type['Synaptic Delay'].iloc[0]
            conn_prob = synapse_type['Connection Probability'].iloc[0]

            S = get_synapses(neurons[connection['pre_idx']]['neuron'], neurons[connection['post_idx']]['neuron'], tau_d, tau_r, tau_f, A_SE, U_SE, t_delay, 1) # 1 -> conn_prob
            
            ready_synapses.append(S)
            
            print(f'Pre neuron of type {pre_type} and post neuron of type {post_type} are connected...')
'''
print('All neurons within CA1 are connected')

# example_list = [NeuronTypes['CA1 Pyramidal'], NeuronTypes['CA1 Bistratified']]
# example = example_list[0]
# 



# for neuron in neurons:
#     net.add(neuron['neuron'])

# net.add(ready_neurons)



# for i in range(len(example_list)):
#     #example = example_list[i]
#     synapse = synapse_types[(synapse_types['Presynaptic Neuron Type'] == 'EC LIII Pyramidal') & (synapse_types['Postsynaptic Neuron Type'] == 'CA1 Pyramidal')]
#     tau_d = synapse['tau_d'].iloc[0]
#     tau_r = synapse['tau_r'].iloc[0]
#     tau_f = synapse['tau_f'].iloc[0]
#     A_SE = synapse['g'].iloc[0]
#     U_SE = synapse['u'].iloc[0]
#     t_delay = synapse['Synaptic Delay'].iloc[0]
#     conn_prob = 1 #synapse['Connection Probability'].iloc[0]
#     S = get_synapses(lec, example_list[i], tau_d, tau_r, tau_f, A_SE, U_SE, t_delay, conn_prob)
#     ready_synapses.append(S)
#     #net.add(S)




# G = example_list[0]


# Adding a input from generators for every neuron in net
# for neuron in neurons:
#     type = neuron['type']
#     for generator in ["EC LIII Pyramidal", 'CA3 Pyramidal']:
#         synapse = synapse_types[(synapse_types['Presynaptic Neuron Type'] == generator) & (synapse_types['Postsynaptic Neuron Type'] == neuron['type'])]
        
#         if not synapse.empty: # if there is such type in the table:
#             tau_d, tau_r, tau_f = synapse['tau_d'].iloc[0], synapse['tau_r'].iloc[0], synapse['tau_f'].iloc[0]
#             A_SE, U_SE = synapse['g'].iloc[0], synapse['u'].iloc[0]
#             t_delay = synapse['Synaptic Delay'].iloc[0]
#             conn_prob = 1 #synapse['Connection Probability'].iloc[0]
#             if generator == "EC LIII Pyramidal":
#                 print(f'Connecting generator to neuron {type}')
#                 S = get_synapses(mec, neuron['neuron'], tau_d, tau_r, tau_f, A_SE, U_SE, t_delay, conn_prob)
#                 ready_synapses.append(S)
#                 S = get_synapses(lec, neuron['neuron'], tau_d, tau_r, tau_f, A_SE, U_SE, t_delay, conn_prob)
#                 ready_synapses.append(S)
#             else: 
#                 S = get_synapses(ca3, neuron['neuron'], tau_d, tau_r, tau_f, A_SE, U_SE, t_delay, conn_prob)
#                 ready_synapses.append(S)
#             print(f'Generator is connected to {type}...')

# net.add(ready_synapses)

# print('Inputs from generators are added')

# for neuron in neurons:
#     if neuron['type'] == 'CA1 Pyramidal': 
#         print(neuron['neuron'])
#         G = neuron['neuron']
#         M = StateMonitor(G, 'v', record=True)
#         break

# M = StateMonitor(G, 'v', record=True) 
# net.add(M)
# net.run(tfinal)
# plot(M.t/ms, M.v[0])
# plt.show()




#'''

#'''

'''
S = Synapses(
    N,
    N,
    "w : 1",
    on_pre={"up": "I += w", "down": "I -= w"},
    delay={"up": 0 * ms, "down": 1 * ms},
)
S.connect()
S.w[:] = weights.flatten()

N_exc.run_regularly("I_noise = 5*randn()", dt=1 * ms)
N_inh.run_regularly("I_noise = 2*randn()", dt=1 * ms)

run(tfinal)

fig, (ax, ax_voltage) = plt.subplots(2, 1, sharex=True,
                                     gridspec_kw={'height_ratios': (3, 1)})

ax.scatter(spikemon.t / ms, spikemon.i[:], marker="_", color="k", s=10)
ax.set_xlim(0, tfinal / ms)
ax.set_ylim(0, len(N))
ax.set_ylabel("neuron number")
ax.set_yticks(np.arange(0, len(N), 100))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.axhline(Ne, color="k")
ax.text(500, 900, 'inhibitory', backgroundcolor='w', color='k', ha='center')
ax.text(500, 400, 'excitatory', backgroundcolor='w', color='k', ha='center')

ax_voltage.plot(statemon.t / ms, np.clip(statemon.v[0], -np.inf, 30),
               color='k')
ax_voltage.text(25, 0, 'v₁(t)')
ax_voltage.set_xticks(np.arange(0, tfinal / ms, 100))
ax_voltage.spines['right'].set_visible(False)
ax_voltage.spines['top'].set_visible(False)
ax_voltage.set_xlabel("time, ms")

plt.show()

'''

#'''
mec = []
with (open("presimulation_files/mec_generators.pickle", "rb")) as openfile:
    while True:
        try:
            mec = pickle.load(openfile)
        except EOFError:
            break

print('mec: \n', mec)

lec = []
with (open("presimulation_files/lec_generators.pickle", "rb")) as openfile:
    while True:
        try:
            lec = pickle.load(openfile)
        except EOFError:
            break

print('lec: \n', lec)

ca3_gen = []
with (open("presimulation_files/ca3_generators.pickle", "rb")) as openfile:
    while True:
        try:
            ca3_gen = pickle.load(openfile)
        except EOFError:
            break

print('ca3: \n', ca3_gen)

'''
types = []
for pop in populations :
    if pop[1] not in types:
        types.append(pop[1])

#populations.extend([[ca1, 'ca1'], [ca3, 'ca3'], [ec3, 'ec3']])

ecw = []
caw = []
necw = []
ncaw = []

for type in types:
    if type not in ecw:
        necw.append(type)
    if type not in caw:
        ncaw.append(type)
print('Types that have no input from ec3:', necw)
print('Types that have no input from ca3:', ncaw)


for index, synapse in synapse_types.iterrows():
    pass
    #if synapse['Presynaptic Neuron Type'] == 'CA1 Pyramidal' == synapse['Postsynaptic Neuron Type']:
    #print(synapse['Presynaptic Neuron Type'], synapse['Postsynaptic Neuron Type'])
    # tau_d, tau_r, tau_f = synapse['tau_d'], synapse['tau_r'], synapse['tau_f']
    # A_SE, U_SE = synapse['g'], synapse['u']
    # t_delay = synapse['Synaptic Delay']
    # conn_prob = synapse['Connection Probability']
    # S = get_synapses(Groups[synapse['Presynaptic Neuron Type']], Groups[synapse['Postsynaptic Neuron Type']], tau_d, tau_r, tau_f, A_SE, U_SE, t_delay, conn_prob)
    
    # if synapse['Presynaptic Neuron Type'] == 'CA1 Pyramidal': 
    #     S = get_synapses(ca1, Groups[synapse['Postsynaptic Neuron Type']], tau_d, tau_r, tau_f, A_SE, U_SE, t_delay, conn_prob)
    # if synapse['Presynaptic Neuron Type'] == 'CA3 Pyramidal': 
    #     S = get_synapses(ca3, Groups[synapse['Postsynaptic Neuron Type']], tau_d, tau_r, tau_f, A_SE, U_SE, t_delay, conn_prob)
    # if synapse['Presynaptic Neuron Type'] == 'EC LIII Pyramidal': 
    #     S = get_synapses(ec3, Groups[synapse['Postsynaptic Neuron Type']], tau_d, tau_r, tau_f, A_SE, U_SE, t_delay, conn_prob)
'''


def get_neuron(type):

    eqs = """dv/dt = ((k*(v - vr)*(v - vt) - u_p + I + I_noise + 10*I_syn)/Cm)/ms : 1
         du_p/dt = (a*(b*(v-vr) - u_p))/ms  : 1
         I : 1
         I_noise : 1
         I_syn : 1
         a : 1
         b : 1
         d : 1
         Cm : 1
         k : 1
         vt : 1
         vr : 1
       """

    if type == 'CA1 Axo-Axonic': type = 'CA1 Axo-axonic'
    if type == 'CA1 Horizontal Axo-Axonic': type = 'CA1 Horizontal Axo-axonic'
    neuron_param = neuron_types[neuron_types['Neuron Type'] == type]
    Vpeak = neuron_param['Izh Vpeak'].iloc[0]
    Vmin = neuron_param['Izh Vmin'].iloc[0]
    refr = neuron_param['Refractory Period'].iloc[0]
    Group = NeuronGroup(1, eqs, threshold="v>=Vpeak", reset="v=Vmin; u_p+=d", refractory= refr, method="euler") #neuron['Population Size']
    Group.a = neuron_param['Izh a'].iloc[0]
    Group.b = neuron_param['Izh b'].iloc[0]
    Group.Cm = neuron_param['Izh C'].iloc[0]
    Group.d = neuron_param['Izh d'].iloc[0]
    Group.vr = neuron_param['Izh Vr'].iloc[0]
    Group.vt = neuron_param['Izh Vt'].iloc[0]
    Group.k = neuron_param['Izh k'].iloc[0]
    Group.I = 0
    Group.I_noise = 0
    return Group