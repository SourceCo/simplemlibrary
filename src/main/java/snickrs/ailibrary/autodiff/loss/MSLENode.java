package snickrs.ailibrary.autodiff.loss;

import snickrs.ailibrary.autodiff.*;
import snickrs.ailibrary.autodiff.math.*;

public class MSLENode extends UnaryNode {
	public MSLENode(Node target, Node prediction) {
		super(create(target, prediction));
	}
	public static Node create(Node target, Node prediction) {
		return new MeanNode(new ExponentNode(new SubtractionNode(new Log1pNode(target), new Log1pNode(prediction)), new ConstantNode(2.0f)));
	}
}
