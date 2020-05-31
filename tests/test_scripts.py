
import unittest
import mock
import mdciao.examples

class Test_ExamplesCLTs(unittest.TestCase):

    # It's already a test!
    def setUp(self):
        self.xCLTs = mdciao.examples.ExamplesCLTs(
            test=True
        )


    def test_mdc_sites(self):
        input_values = (val for val in ["3", "3","3","3"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            self.xCLTs.run("mdc_sites", write_to_tmpdir=True)

    def test_mdc_neighborhood(self):
        #input_values = (val for val in ["3", "3"])
        #with mock.patch('builtins.input', lambda *x: next(input_values)):
        self.xCLTs.run("mdc_neighborhoods", write_to_tmpdir=True)

    def test_mdc_interface(self):
        #input_values = (val for val in ["3", "3"])
        #with mock.patch('builtins.input', lambda *x: next(input_values)):
        self.xCLTs.run("mdc_interface", write_to_tmpdir=True)





if __name__ == '__main__':
    unittest.main()