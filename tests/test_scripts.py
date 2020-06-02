
import unittest
import mdciao.examples
from tempfile import TemporaryDirectory as _TDir


class Test_ExamplesCLTs(unittest.TestCase):

    # It's already a test!
    def setUp(self):
        self.xCLTs = mdciao.examples.ExamplesCLTs(
            test=True
        )

    def test_mdc_sites(self):
        with  _TDir("_mdciao_tests") as od:
            CP = self.xCLTs.run("mdc_sites", show=False, output_dir=od)
            assert CP.returncode==0

    def test_mdc_neighborhood(self):
        #input_values = (val for val in ["3", "3"])
        #with mock.patch('builtins.input', lambda *x: next(input_values)):
        with  _TDir("_mdciao_tests") as od:
            CP = self.xCLTs.run("mdc_neighborhoods", show=False, output_dir=od)
            assert CP.returncode==0


    def test_mdc_interface(self):
        with  _TDir("_mdciao_tests") as od:
            CP = self.xCLTs.run("mdc_interface", show=False, output_dir=od)
            assert CP.returncode==0



if __name__ == '__main__':
    unittest.main()