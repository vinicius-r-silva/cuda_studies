subbsystem:
	cd number_sequencies && $(MAKE)
	cd device_properties && $(MAKE)
	cd edge_detection && $(MAKE)
	cd imgFlip && $(MAKE)

numbers:
	cd number_sequencies && $(MAKE)

edge_detection:
	cd edge_detection && $(MAKE)

imgFlip:
	cd number_sequencies && $(MAKE)

clean:
	cd number_sequencies && $(MAKE) clean
	cd device_properties && $(MAKE) clean
	cd edge_detection && $(MAKE) clean
	cd imgFlip && $(MAKE) clean

	