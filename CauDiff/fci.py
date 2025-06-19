from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils
import numpy as np

data = np.random.rand(1000, 3)
g, edges = fci(data)
pdy = GraphUtils.to_pydot(g)
pdy.write_png('simple_test.png')