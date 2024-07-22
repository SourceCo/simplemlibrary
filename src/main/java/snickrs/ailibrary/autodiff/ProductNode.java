package snickrs.ailibrary.autodiff;

import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;

public class ProductNode extends Node {
	public ProductNode(Node ... children) {
		super(children);
	}
	public ProductNode(List<Node> children) {
		super(children);
	}
	public INDArray child_evaluate() {
		ArrayList<INDArray> evals = get_evals();
		if(children.size() > 2) {
			this.m = evals.get(0);
			for(int i = 1; i < evals.size(); i++) {
				this.m = this.m.mul(evals.get(i));
			}
		} else {
			this.m = children.get(0).m.mul(children.get(1).m);	
		}
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		if(children.size() <= 2) {
			partials.set(0, unbroadcast(children.get(0).m, upstream.mul(children.get(1).m)));
			partials.set(1, unbroadcast(children.get(1).m, upstream.mul(children.get(0).m)));
		} else {
			for(int i = 0; i < children.size(); i++) {
				INDArray product = this.m.div(children.get(i).m);
				partials.set(i, unbroadcast(children.get(i).m, upstream.mul(product)));
			}
		}
	}
    public long[] shape() {
        long[] resultShape = children.get(0).shape();
        for (int i = 1; i < children.size(); i++) {
            long[] shape = children.get(i).shape();
            resultShape = broadcast(resultShape, shape);
        }
        return resultShape;
    }
}
