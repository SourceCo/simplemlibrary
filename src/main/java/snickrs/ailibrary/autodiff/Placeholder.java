package snickrs.ailibrary.autodiff;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Placeholder extends Node {
	public long[] dims;
	public Placeholder (String name, long ... dims) {
		this.name = name;
		this.dims = dims;
		boolean instantiate = true;
		for(long x : dims) {
			if(x < 0) {
				instantiate = false;
				break;
			}
		}
		if(instantiate) this.m = Nd4j.zeros(dims);
	}
	public boolean isConstant(Node val) {
		return val.hashCode() != this.hashCode();
	}
	public void set(INDArray m) {
		this.m = m;
	}
	@Override
	public String toString() {
		return this.name;
	}
	@Override
	public INDArray child_evaluate() {
		return this.m;
	}
	public long[] shape() {
		return this.m.shape();
	}
}
