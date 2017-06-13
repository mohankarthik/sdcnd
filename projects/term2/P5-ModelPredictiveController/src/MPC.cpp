/* ==================== INCLUDES ==================== */
#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

/* =================== NAMESPACES =================== */
using CppAD::AD;

/* ==================== DEFINES ===================== */
/** SIMULATOR MACROS **/
#define LF              (2.67)
#define STEER_LIMIT     (0.436332)
#define THROTTLE_LIMIT  (1.0)

/** MPC MACROS **/
#define N               (10)
#define DT              (0.10)
#define NUM_VARS        (6)

/** WEIGHTS **/
#define W_CTE           (1000.0)
#define W_EPSI          (1000.0)
#define W_V             (0.01)
#define W_STEER         (1.0)
#define W_THROTTLE      (1.0)
#define W_STEER_DIFF    (1.0)
#define W_THROTTLE_DIFF (1.0)

/** REFERENCE VALUES **/
#define REF_CTE         (0.0)
#define REF_EPSI        (0.0)
#define REF_V           (200.0)

/** MISC **/
#define INF             (1.0e19)

/** LPOPT MACROS **/
#define X_START         (0)
#define Y_START         (X_START + N)
#define PSI_START       (Y_START + N)
#define V_START         (PSI_START + N)
#define CTE_START       (V_START + N)
#define EPSI_START      (CTE_START + N)
#define STEER_START     (EPSI_START + N)
#define THROTTLE_START  (STEER_START + N - 1)

/** DERIVED MACROS **/
#define TOT_VARS        ((NUM_VARS * N) + (2 * (N - 1)))
#define TOT_CONSTRAINTS (N * NUM_VARS)

/* ================= FG EVAL ================== */
class FG_eval 
{
  public:
    /* ================= DATA TYPES ================== */
    /*!
     * Typecast for easy use
     */
    typedef CPPAD_TESTVECTOR(AD<double>) ADvector;

    /* ==================== DATA ===================== */
    /*
    /*! 
     * Fitted polynomial coefficients
     */
    Eigen::VectorXd vCoeffs;

    /* ==================== CODE ===================== */
    /*!
     * Constructor, saves the coefficients
     */
    FG_eval(Eigen::VectorXd vCoeffs) 
    { 
      this->vCoeffs = vCoeffs; 
    }

    void operator()(ADvector& fg, const ADvector& vars) 
    {
      /* Initialize the cost to 0 */
      fg[0] = 0;

      /* The part of the cost based on the reference state */
      for (int32_t i = 0; i < N; i++) 
      {
        fg[0] += W_CTE  * CppAD::pow(vars[CTE_START + i]  - REF_CTE,  2);
        fg[0] += W_EPSI * CppAD::pow(vars[EPSI_START + i] - REF_EPSI, 2);
        fg[0] += W_V    * CppAD::pow(vars[V_START + i]    - REF_V,    2);
      }

      /* Minimize the use of actuators */
      for (int32_t i = 0; i < N - 1; i++) 
      {
        fg[0] += W_STEER    * CppAD::pow(vars[STEER_START + i],    2);
        fg[0] += W_THROTTLE * CppAD::pow(vars[THROTTLE_START + i], 2);
      }

      /* Minimize the value gap between sequential actuations,
      for smoother actuations */
      for (int32_t i = 0; i < N - 2; i++) 
      {
        fg[0] += W_STEER_DIFF    * CppAD::pow(vars[STEER_START + i + 1]    - vars[STEER_START + i],    2);
        fg[0] += W_THROTTLE_DIFF * CppAD::pow(vars[THROTTLE_START + i + 1] - vars[THROTTLE_START + i], 2);
      }

      /* Initial constraints, this has to be the same as the past value */
      fg[1 + X_START] = vars[X_START];
      fg[1 + Y_START] = vars[Y_START];
      fg[1 + PSI_START] = vars[PSI_START];
      fg[1 + V_START] = vars[V_START];
      fg[1 + CTE_START] = vars[CTE_START];
      fg[1 + EPSI_START] = vars[EPSI_START];

      /* The rest of the constraints, have to follow the
      vehicle model */
      for (int32_t i = 0; i < N - 1; i++) 
      {
        /* The state at time t+1 */
        const AD<double> x1     = vars[X_START + i + 1];
        const AD<double> y1     = vars[Y_START + i + 1];
        const AD<double> psi1   = vars[PSI_START + i + 1];
        const AD<double> v1     = vars[V_START + i + 1];
        const AD<double> cte1   = vars[CTE_START + i + 1];
        const AD<double> epsi1  = vars[EPSI_START + i + 1];

        /* The state at time t */
        const AD<double> x0     = vars[X_START + i];
        const AD<double> y0     = vars[Y_START + i];
        const AD<double> psi0   = vars[PSI_START + i];
        const AD<double> v0     = vars[V_START + i];
        const AD<double> cte0   = vars[CTE_START + i];
        const AD<double> epsi0  = vars[EPSI_START + i];

        /* Considering the actuation at time t */
        const AD<double> steer0     = vars[STEER_START + i];
        const AD<double> throttle0  = vars[THROTTLE_START + i];

        const AD<double> f0 = vCoeffs[0] + (vCoeffs[1] * x0) + (vCoeffs[2] * CppAD::pow(x0, 2)) + (vCoeffs[3] * CppAD::pow(x0, 3));
        const AD<double> psides0 = CppAD::atan(vCoeffs[1] + (2.0 * vCoeffs[2] * x0) + (3.0 * vCoeffs[3] * CppAD::pow(x0, 2)));

        /* Now constrain the values such that they all fit the vehicle model */
        const auto dMultiplier = (steer0 / LF);
        fg[2 + X_START + i] = x1 - (x0 + (v0 * CppAD::cos(psi0) * DT));
        fg[2 + Y_START + i] = y1 - (y0 + (v0 * CppAD::sin(psi0) * DT));
        fg[2 + PSI_START + i] = psi1 - (psi0 - (v0 * dMultiplier * DT));
        fg[2 + V_START + i] = v1 - (v0 + (throttle0 * DT));
        fg[2 + CTE_START + i] = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * DT));
        fg[2 + EPSI_START + i] = epsi1 - ((psi0 - psides0) - (v0 * dMultiplier * DT));
      }
    }
};

/* ================= MPC ================== */
MPC::MPC() {}
MPC::~MPC() {}

/*!
 * Solves the MPC equation
 */
vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) 
{
  bool bOK = true;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  Dvector vVars(TOT_VARS);
  Dvector vLowerBoundVars(TOT_VARS);
  Dvector vUpperBoundVars(TOT_VARS);
  Dvector vLowerBoundConstraints(TOT_CONSTRAINTS);
  Dvector vUpperBoundConstraints(TOT_CONSTRAINTS);

  CppAD::ipopt::solve_result<Dvector> vSolution;

  /* Extract the states */
  const double &dX    = state[0];
  const double &dY    = state[1];
  const double &dPsi  = state[2];
  const double &dV    = state[3];
  const double &dCte  = state[4];
  const double &dEpsi = state[5];

  /* Initial value of the independent variables.
  SHOULD BE 0 besides initial state */
  
  for (int32_t i = 0; i < TOT_VARS; i++) 
  {
    vVars[i] = 0;
  }
  
  /* Set the initial variable values */
  vVars[X_START]    = dX;
  vVars[Y_START]    = dY;
  vVars[PSI_START]  = dPsi;
  vVars[V_START]    = dV;
  vVars[CTE_START]  = dCte;
  vVars[EPSI_START] = dEpsi;

  /* Set the values to max for all variables
  other than the actuators */
  for (int32_t i = 0; i < STEER_START; i++) 
  {
    vLowerBoundVars[i] = -INF;
    vUpperBoundVars[i] = INF;
  }

  /* The upper and lower limits of delta are set to -25 and 25
   degrees (values in radians) */
  for (int32_t i = STEER_START; i < THROTTLE_START; i++) 
  {
    vLowerBoundVars[i] = -STEER_LIMIT * LF;
    vUpperBoundVars[i] = STEER_LIMIT  * LF;
  }

  // Acceleration/decceleration upper and lower limits.
  for (int32_t i = THROTTLE_START; i < TOT_VARS; i++) 
  {
    vLowerBoundVars[i] = -THROTTLE_LIMIT;
    vUpperBoundVars[i] = THROTTLE_LIMIT;
  }

  /* Lower and upper limits for the constraints
   Should be 0 besides initial state */
  for (int32_t i = 0; i < TOT_CONSTRAINTS; i++) 
  {
    vLowerBoundConstraints[i] = 0;
    vUpperBoundConstraints[i] = 0;
  }

  /* Set the initial constraints to be the past value */
  vLowerBoundConstraints[X_START]     = dX;
  vLowerBoundConstraints[Y_START]     = dY;
  vLowerBoundConstraints[PSI_START]   = dPsi;
  vLowerBoundConstraints[V_START]     = dV;
  vLowerBoundConstraints[CTE_START]   = dCte;
  vLowerBoundConstraints[EPSI_START]  = dEpsi;

  vUpperBoundConstraints[X_START]     = dX;
  vUpperBoundConstraints[Y_START]     = dY;
  vUpperBoundConstraints[PSI_START]   = dPsi;
  vUpperBoundConstraints[V_START]     = dV;
  vUpperBoundConstraints[CTE_START]   = dCte;
  vUpperBoundConstraints[EPSI_START]  = dEpsi;

  /* Create the objective function */
  FG_eval fg_eval(coeffs);

  /* Options for IPOPT solver */
  std::string options;
  options += "Integer print_level  0\n";
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  options += "Numeric max_cpu_time          0.5\n";

  /* solve the problem */
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vVars, vLowerBoundVars, vUpperBoundVars, vLowerBoundConstraints,
      vUpperBoundConstraints, fg_eval, vSolution);

  /* Check some of the solution values */
  bOK &= vSolution.status == CppAD::ipopt::solve_result<Dvector>::success;

  /* Calculate the cost & print it out */
  const auto dCost = vSolution.obj_value;
  std::cout << "Cost " << dCost << std::endl;

  /* Save the solution */
  gvSolX.clear();
  for (int i = 0; i < vSolution.x.size(); ++i) 
  {
    gvSolX.push_back(vSolution.x[i]);
  }

  /* Return just the two most immediate actuations */
  return {vSolution.x[STEER_START], vSolution.x[THROTTLE_START]};
}

/*!
 * Get the X co-ordinate tragectory
 */
vector<double> MPC::getXs() 
{
  return vector<double>(gvSolX.begin() + X_START, gvSolX.begin() + (X_START + N));
}

/*!
 * Get the Y co-ordinate tragectory
 */
vector<double> MPC::getYs() 
{
  return vector<double>(gvSolX.begin() + Y_START, gvSolX.begin() + (Y_START + N));
}
