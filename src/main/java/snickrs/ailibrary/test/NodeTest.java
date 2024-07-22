package snickrs.ailibrary;
import snickrs.ailibrary.autodiff.*;
import snickrs.ailibrary.autodiff.activation.*;
import snickrs.ailibrary.autodiff.cnn.*;
import snickrs.ailibrary.autodiff.constraints.*;
import snickrs.ailibrary.autodiff.dropout.*;
import snickrs.ailibrary.autodiff.loss.*;
import snickrs.ailibrary.autodiff.math.*;
import snickrs.ailibrary.autodiff.merge.*;
import snickrs.ailibrary.autodiff.pooling.*;
import snickrs.ailibrary.autodiff.nn.*;
import snickrs.ailibrary.autodiff.normalization.*;
import snickrs.ailibrary.autodiff.optimization.*;
import snickrs.ailibrary.autodiff.regularization.*;
import snickrs.ailibrary.autodiff.reshape.*;
import snickrs.ailibrary.autodiff.weights.*;

import snickrs.ailibrary.datasets.*;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.enums.WeightsFormat;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import org.threeten.bp.LocalDateTime;
import org.threeten.bp.format.DateTimeFormatter;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NodeTest {
	public static void main(String[] args) {
		Variable t0 = new Variable("t0", 10, 2);
		System.out.println(t0.m);
		t0.expand(0);
		System.out.println(t0.m);
	}
}
