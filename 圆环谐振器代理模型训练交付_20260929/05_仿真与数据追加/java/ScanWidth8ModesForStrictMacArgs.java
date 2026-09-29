import com.comsol.model.*;
import com.comsol.model.util.*;

public class ScanWidth8ModesForStrictMacArgs {
  private static final int REQUESTED_EIGENFREQUENCIES = 10;
  private static final int MAX_SOLUTION_INDEX = 20;
  private static final int N_THETA = 64;
  private static final double FREQ_CENTER_MHZ = 10.0;

  private static double parseNumber(String text) {
    if (text.startsWith("m")) {
      return -Double.parseDouble(text.substring(1));
    }
    return Double.parseDouble(text);
  }

  private static String clean(String text) {
    if (text == null) {
      return "";
    }
    return text.replace(",", ";").replace("\n", " ").replace("\r", " ");
  }

  private static double[][] evalAt(Model model, int modeIndex, double xUm, double yUm, double zUm) {
    try {
      model.result().numerical().remove("interp_scan");
    } catch (Exception ignored) {
    }
    model.result().numerical().create("interp_scan", "Interp");
    model.result().numerical("interp_scan").set("data", "dset1");
    model.result().numerical("interp_scan").set("solvertype", "solnum");
    model.result().numerical("interp_scan").set("solnum", modeIndex);
    model.result().numerical("interp_scan").set("expr", new String[] {"u", "v", "w"});
    model.result().numerical("interp_scan").set("unit", new String[] {"um", "um", "um"});
    model.result().numerical("interp_scan").set("coord", new double[][] {
        new double[] {xUm},
        new double[] {yUm},
        new double[] {zUm}
    });
    return model.result().numerical("interp_scan").getReal();
  }

  private static double[] evalGlobal(Model model, int modeIndex) {
    try {
      model.result().numerical().remove("gev_scan");
    } catch (Exception ignored) {
    }
    model.result().numerical().create("gev_scan", "EvalGlobal");
    model.result().numerical("gev_scan").set("data", "dset1");
    model.result().numerical("gev_scan").set("innerinput", "manual");
    model.result().numerical("gev_scan").set("solnum", "1");
    model.result().numerical("gev_scan").set("solnumindices", new int[] {modeIndex});
    model.result().numerical("gev_scan").set("expr", new String[] {"solid.Q_eig", "freq"});
    model.result().numerical("gev_scan").set("unit", new String[] {"1", "MHz"});
    double[][] values = model.result().numerical("gev_scan").getReal();
    double q = values.length > 0 && values[0].length > 0 ? values[0][0] : Double.NaN;
    double f = values.length > 1 && values[1].length > 0 ? values[1][0] : Double.NaN;
    return new double[] {q, f};
  }

  private static void configureEigen(Model model, String studyTag) {
    model.study(studyTag).feature("eig").set("neigs", String.valueOf(REQUESTED_EIGENFREQUENCIES));
    model.study(studyTag).feature("eig").set("shift", String.valueOf(FREQ_CENTER_MHZ));
    try {
      model.sol("sol1").feature("e1").set("neigs", String.valueOf(REQUESTED_EIGENFREQUENCIES));
      model.sol("sol1").feature("e1").set("shift", String.valueOf(FREQ_CENTER_MHZ));
    } catch (Exception ignored) {
    }
  }

  private static void printMode(Model model, int designIndex, int modeIndex, double radiusUm, double zUm) {
    double[] qf = evalGlobal(model, modeIndex);
    double sum = 0.0;
    double sumSq = 0.0;
    int positive = 0;
    int negative = 0;
    double urMin = Double.POSITIVE_INFINITY;
    double urMax = Double.NEGATIVE_INFINITY;
    StringBuilder vector = new StringBuilder();

    for (int i = 0; i < N_THETA; i++) {
      double theta = 2.0 * Math.PI * i / N_THETA;
      double x = radiusUm * Math.cos(theta);
      double y = radiusUm * Math.sin(theta);
      double[][] values = evalAt(model, modeIndex, x, y, zUm);
      double u = values.length > 0 && values[0].length > 0 ? values[0][0] : Double.NaN;
      double v = values.length > 0 && values[0].length > 1 ? values[0][1] : Double.NaN;
      double w = values.length > 0 && values[0].length > 2 ? values[0][2] : Double.NaN;
      double ur = u * Math.cos(theta) + v * Math.sin(theta);
      if (vector.length() > 0) {
        vector.append(';');
      }
      vector.append(u).append(';').append(v).append(';').append(w);
      sum += ur;
      sumSq += ur * ur;
      if (ur >= 0) {
        positive++;
      } else {
        negative++;
      }
      urMin = Math.min(urMin, ur);
      urMax = Math.max(urMax, ur);
    }

    double urMean = sum / N_THETA;
    double urRms = Math.sqrt(sumSq / N_THETA);
    double sameSignFraction = Math.max(positive, negative) / (double) N_THETA;
    double breathingUniformity = urRms > 0 ? Math.abs(urMean) / urRms : Double.NaN;
    int passes = sameSignFraction >= 0.95 && breathingUniformity >= 0.95 ? 1 : 0;

    System.out.println("MODE," + designIndex
        + "," + modeIndex
        + "," + qf[1]
        + "," + qf[0]
        + "," + sameSignFraction
        + "," + breathingUniformity
        + "," + urMean
        + "," + urRms
        + "," + urMin
        + "," + urMax
        + "," + passes
        + "," + vector);
  }

  public static void main(String[] args) throws Exception {
    if (args.length < 19 || ((args.length - 3) % 18) != 0) {
      System.err.println("Usage: ScanWidth8ModesForStrictMacArgs <model.mph> <studyTag> <z_um> <a_um> <h_um> <C1_um> <S1_um> ... <C8_um> <S8_um> [repeat...]");
      return;
    }

    String modelPath = args[0];
    String studyTag = args[1];
    double zUm = parseNumber(args[2]);
    int designCount = (args.length - 3) / 18;

    ModelUtil.initStandalone(false);
    Model model = ModelUtil.load("model", modelPath);
    System.out.println("DESIGN,index,a_ring_um,h_ring_um,C1_um,S1_um,C2_um,S2_um,C3_um,S3_um,C4_um,S4_um,C5_um,S5_um,C6_um,S6_um,C7_um,S7_um,C8_um,S8_um,status,message");
    System.out.println("MODE,index,mode_index,freq_MHz,Q,same_sign_fraction,breathing_uniformity,ur_mean_um,ur_rms_um,ur_min_um,ur_max_um,passes_breathing_check,vector");

    for (int i = 0; i < designCount; i++) {
      int designIndex = i + 1;
      int base = 3 + 18 * i;
      double aUm = parseNumber(args[base]);
      double hUm = parseNumber(args[base + 1]);
      double c1 = parseNumber(args[base + 2]);
      double s1 = parseNumber(args[base + 3]);
      double c2 = parseNumber(args[base + 4]);
      double s2 = parseNumber(args[base + 5]);
      double c3 = parseNumber(args[base + 6]);
      double s3 = parseNumber(args[base + 7]);
      double c4 = parseNumber(args[base + 8]);
      double s4 = parseNumber(args[base + 9]);
      double c5 = parseNumber(args[base + 10]);
      double s5 = parseNumber(args[base + 11]);
      double c6 = parseNumber(args[base + 12]);
      double s6 = parseNumber(args[base + 13]);
      double c7 = parseNumber(args[base + 14]);
      double s7 = parseNumber(args[base + 15]);
      double c8 = parseNumber(args[base + 16]);
      double s8 = parseNumber(args[base + 17]);
      String status = "ok";
      String message = "";

      System.out.println("RUNNING," + designIndex + "," + aUm + "," + hUm + "," + c1 + "," + s1 + "," + c2 + "," + s2 + "," + c3 + "," + s3 + "," + c4 + "," + s4 + "," + c5 + "," + s5 + "," + c6 + "," + s6 + "," + c7 + "," + s7 + "," + c8 + "," + s8);
      try {
        model.param().set("a_ring", aUm + "[um]");
        model.param().set("h_ring", hUm + "[um]");
        model.param().set("C1", c1 + "[um]");
        model.param().set("S1", s1 + "[um]");
        model.param().set("C2", c2 + "[um]");
        model.param().set("S2", s2 + "[um]");
        model.param().set("C3", c3 + "[um]");
        model.param().set("S3", s3 + "[um]");
        model.param().set("C4", c4 + "[um]");
        model.param().set("S4", s4 + "[um]");
        model.param().set("C5", c5 + "[um]");
        model.param().set("S5", s5 + "[um]");
        model.param().set("C6", c6 + "[um]");
        model.param().set("S6", s6 + "[um]");
        model.param().set("C7", c7 + "[um]");
        model.param().set("S7", s7 + "[um]");
        model.param().set("C8", c8 + "[um]");
        model.param().set("S8", s8 + "[um]");
        model.component("comp1").geom("geom1").run();
        model.component("comp1").mesh("mesh1").run();
        configureEigen(model, studyTag);
        model.study(studyTag).run();
        System.out.println("DESIGN," + designIndex + "," + aUm + "," + hUm + "," + c1 + "," + s1 + "," + c2 + "," + s2 + "," + c3 + "," + s3 + "," + c4 + "," + s4 + "," + c5 + "," + s5 + "," + c6 + "," + s6 + "," + c7 + "," + s7 + "," + c8 + "," + s8 + "," + status + "," + message);
        for (int modeIndex = 1; modeIndex <= MAX_SOLUTION_INDEX; modeIndex++) {
          try {
            printMode(model, designIndex, modeIndex, aUm, zUm);
          } catch (Exception ignored) {
          }
        }
      } catch (Exception e) {
        status = "error";
        message = clean(e.getMessage());
        System.out.println("DESIGN," + designIndex + "," + aUm + "," + hUm + "," + c1 + "," + s1 + "," + c2 + "," + s2 + "," + c3 + "," + s3 + "," + c4 + "," + s4 + "," + c5 + "," + s5 + "," + c6 + "," + s6 + "," + c7 + "," + s7 + "," + c8 + "," + s8 + "," + status + "," + message);
      }
    }
  }
}
