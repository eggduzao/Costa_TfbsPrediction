package org.blacksmith.suite;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.blacksmith.BlacksmithInstrumentedTests;

@RunWith(Suite.class)
@Suite.SuiteClasses({BlacksmithInstrumentedTests.class})
public class BlacksmithInstrumentedTestSuite {}
