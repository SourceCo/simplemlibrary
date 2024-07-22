package snickrs.ailibrary.autodiff;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ConstantNode extends Node {
	public double constant;
	public ConstantNode(float constant) {
		this.constant = constant;
		this.m = Nd4j.scalar(constant);
	}
	@Override
	public String toString() {
		return Double.toString(constant);
	}
	public boolean isConstant(Node val) {
		return true;
	}
	public boolean isConstant() {
		return true;
	}
	public boolean equals(double val) {
		return this.constant == val;
	}
	public boolean isInteger() {
		return this.constant % 1.0f == 0.0f;
	}
	public INDArray child_evaluate() {
		return this.m;
	}
}
