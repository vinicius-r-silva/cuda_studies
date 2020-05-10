subbsystem:
	cd number_sequencies && $(MAKE)
	cd device_properties && $(MAKE)

numbers:
	cd number_sequencies && $(MAKE)

clean:
	cd number_sequencies && $(MAKE) clean
	cd device_properties && $(MAKE) clean

	