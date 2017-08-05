import sys
import ns.applications
import ns.core
import ns.flow_monitor
import ns.internet
import ns.mobility
import ns.network
import ns.olsr
import ns.wifi
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn import preprocessing
from keras.preprocessing import sequence

try:
   import ns.visualizer
except ImportError:
   pass

DISTANCE = 100 # (m)
NUM_NODES_SIDE = 3

class DQNAgent:
	
      def __init__(self, state_size, actions):
	self.state_size = state_size
	self.action_size = actions
	self.gamma = 0.95
	self.epsilon = 1.0
	self.epsilon_min = 0.01
	self.epsilon_decay = 0.995
	self.learning_rate = 0.001
	self.model = self.build_model()

      def build_model(self):
	self.model = Sequential()
	self.model.add(LSTM(32,input_shape=self.state_size,activation='relu',return_sequence=True))
	self.model.add(LSTM(16,activation='relu',return_sequence=True))
	self.model.add(LSTM(8,activation='relu',return_sequence=True))
	self.model.add(Dense(self.action_size,activation='linear'))
	sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
	self.model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate),metrics=['accuracy'])
	return self.model

      def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
      
      def act(self, state):
        if np.random.rand() <= self.epsilon:
          return random.randrange(self.actions_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    
      def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
	
			

def main(argv):

	cmd = ns.core.CommandLine()
	
	cmd.NumNodesSide = None
	cmd.AddValue("NumNodesSide","Grid side number of nodes (total number of nodes will be this number squared)")
	
	cmd.Results = None
	cmd.AddValue("Results","Write XML results to file")

	cmd.Plot = None
	cmd.AddValue("Plot","Plot the results using the matplotlib python module");

	cmd.Parse(argv)

	wifi = ns.wifi.WifiHelper.Default()
  	wifiMac = ns.wifi.NqosWifiMacHelper.Default()
        wifiPhy = ns.wifi.YansWifiPhyHelper.Default()
	wifiChannel = ns.wifi.YansWifiChannelHelper.Default()
	wifiPhy.SetChannel(wifiChannel.Create())
	ssid = ns.wifi.Ssid("wifi-default")
	wifi.SetRemoteStationManager("ns3::ArfWifiManager")
	wifiMac.SetType("ns3::AdhocWifiMac","Ssid",ns.wifi.SsidValue(ssid))

	internet = ns.internet.InternetStackHelper()
	list_routing = ns.internet.Ipv4ListRoutingHelper()
	olsr_routing = ns.olsr.OlsrHelper()
	static_routing = ns.internet.Ipv4StaticRoutingHelper()
	list_routing.Add(static_routing, 0)
	list_routing.Add(olsr_routing, 100)
	internet.SetRoutingHelper(list_routing)

	ipv4Addresses = ns.internet.Ipv4AddressHelper()
	ipv4Addresses.SetBase(ns.network.Ipv4Address("10.0.0.0"), ns.network.Ipv4Mask("255.255.255.0"))
	
	port = 9
	onOffHelper = ns.applications.OnOffHelper("ns3::UdpSocketFactory",ns.network.Address(ns.network.InetSocketAddress(ns.network.Ipv4Address("10.0.0.1"),port)))
	onOffHelper.SetAttribute("DataRate",ns.network.DataRateValue(ns.network.DataRate("100kbps")))
	onOffHelper.SetAttribute("OnTime",ns.core.StringValue("ns3::ConstantRandomVariable[Constant=1]"))
	onOffHelper.SetAttribute("OffTime",ns.core.StringValue("ns3::ConstantRandomVariable[Constant=0]"))

	addresses = []
	nodes = []

	num_nodes_side = NUM_NODES_SIDE
	for xi in range(num_nodes_side):
	    for yi in range(num_nodes_side):

		node = ns.network.Node()
		nodes.append(node)

		internet.Install(ns.network.NodeContainer(node))

		mobility = ns.mobility.ConstantPositionMobilityModel()
		mobility.SetPosition(ns.core.Vector(xi*DISTANCE, yi*DISTANCE, 0))
		node.AggregateObject(mobility)

		devices = wifi.Install(wifiPhy,wifiMac, node)
		ipv4_interfaces = ipv4Addresses.Assign(devices)
		addresses.append(ipv4_interfaces.GetAddress(0))
	for e in range(500):

		for i,node in enumerate(nodes):
		     destaddr = addresses[(len(addresses)-1-i) % len(addresses)]
		     onOffHelper.SetAttribute("Remote",ns.network.AddressValue(ns.network.InetSocketAddress(destaddr, port)))
		     app = onOffHelper.Install(ns.network.NodeContainer(node))
		     urv = ns.core.UniformRandomVariable()
		     app.Start(ns.core.Seconds(urv.GetValue(20, 30)))

		flowmon_helper = ns.flow_monitor.FlowMonitorHelper()
		monitor = flowmon_helper.InstallAll()
		monitor = flowmon_helper.GetMonitor()
		monitor.SetAttribute("DelayBinWidth",ns.core.DoubleValue(0.001))
		monitor.SetAttribute("JitterBinWidth",ns.core.DoubleValue(0.001))
		monitor.SetAttribute("PacketSizeBinWidth",ns.core.DoubleValue(20))

		ns.core.Simulator.Stop(ns.core.Seconds(44.0))
		ns.core.Simulator.Run()

		monitor.CheckForLostPackets()
		classifier = flowmon_helper.GetClassifier()
		monitor.SerializeToXmlFile("wififlow.xml",True,True)
		flow_stats_inputs = monitor.GetFlowStats()
		flow_array = []
		flow_array.append(flow_stat_inputs.txBytes)
		flow_array.append(flow_stat_inputs.rxBytes)
		flow_array.append(flow_stat_inputs.txPackets)
		flow_array.append(flow_stat_inputs.rxPackets)
		flow_array.append(flow_stat_inputs.lostPackets)
		if st.rxPackets > 0:
		   mean_delay = (flow_stat_inputs.delaySum.GetSeconds() / flow_stat_inputs.rxPackets)	
		   flow_array.append(mean_delay)
		mean_jitter = (flow_stat_inputs.jitterSum.GetSeconds() / (flow_stat_inputs.rxPackets-1))
		flow_array.append(mean_jitter)
		np.array(flow_array)
		df = DataFrame(data=d,index=index)
		min_max_scaler = preprocessing.MinMaxScaler()
		np_scaled = min_max_scaler.fit_transform(df)
		df_normalized = pd.DataFrame(np_scaled)
		X_train = sequence.pad_sequences(df_normalized, 10)
		actor = DQNAgent(X_train.shape[1],num_nodes_side)
		for t in range(500):
			action = actor.act(X_train)
			for i,node in enumerate(nodes):
				destaddr = addresses[action]
				onOffHelper.SetAttribute("Remote",ns.network.AddressValue(ns.network.InetSocketAddress(destaddr, port)))
				app = onOffHelper.Install(ns.network.NodeContainer(node))
				urv = ns.core.UniformRandomVariable()
				app.Start(ns.core.Seconds(urv.GetValue(20, 30)))

			flowmon_helper = ns.flow_monitor.FlowMonitorHelper()
			monitor = flowmon_helper.InstallAll()
			monitor = flowmon_helper.GetMonitor()
			monitor.SetAttribute("DelayBinWidth",ns.core.DoubleValue(0.001))
			monitor.SetAttribute("JitterBinWidth",ns.core.DoubleValue(0.001))
			monitor.SetAttribute("PacketSizeBinWidth",ns.core.DoubleValue(20))

			ns.core.Simulator.Stop(ns.core.Seconds(44.0))
			ns.core.Simulator.Run()

			monitor.CheckForLostPackets()
			classifier = flowmon_helper.GetClassifier()
			monitor.SerializeToXmlFile("wififlow.xml",True,True)
			flow_stats_inputs_next = monitor.GetFlowStats()
			flow_array_next = []
			flow_array_next.append(flow_stat_inputs_next.txBytes)
			flow_array_next.append(flow_stat_inputs_next.rxBytes)
			flow_array_next.append(flow_stat_inputs_next.txPackets)
			flow_array_next.append(flow_stat_inputs_next.rxPackets)
			flow_array_next.append(flow_stat_inputs_next.lostPackets)
			if st.rxPackets > 0:
				mean_delay = (flow_stat_inputs_next.delaySum.GetSeconds() / flow_stat_inputs.rxPackets)	
				flow_array.append(mean_delay)
			mean_jitter = (flow_stat_inputs_next.jitterSum.GetSeconds() / (flow_stat_inputs.rxPackets-1))
			flow_array_next.append(mean_jitter)
			np.array(flow_array_next)
			df_next = DataFrame(data=d,index=index)
			min_max_scaler_next = preprocessing.MinMaxScaler()
			np_scaled_next = min_max_scaler_next.fit_transform(df_next)
			df_normalized_next = pd.DataFrame(np_scaled_next)
			X_next = sequence.pad_sequences(df_normalized_next, 10)
			if(flow_stat_inputs_next.lostPackets > 10):
				reward = -10
			else:
				reward = 20
			actor.remember(X_train,action,X_next)
		        print("episode: {}/{}, score: {}".format(e, 500, t))
		actor.replay(32)
		

	import pylab
	delays = []
	for flow_id,flow_stats in monitor.GetFlowStats():
	    tupl = classifier.FindFlow(flow_id)
	    if tupl.protocol == 17 and tupl.sourcePort == 698:
		continue
	    delays.append(flow_stats.delaySum.GetSeconds()/flow_stats.rxPackets)
	pylab.hist(delays, 20)
	pylab.xlabel("Delay (s)")
	pylab.ylabel("Number of Flows")
	pylab.show()
	return 0

if __name__ == '__main__':
	sys.exit(main(sys.argv))
