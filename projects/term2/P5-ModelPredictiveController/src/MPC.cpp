/* =============================== INCLUDES ================================ */
#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

/* ============================== NAMESPACES =============================== */
using CppAD::AD;

/* ================================= DATA ================================== */
static size_t gStartX     = 0;
static size_t gStartY     = gStartX     + MPC_N;
static size_t gStartPsi   = gStartY     + MPC_N;
static size_t gStartVel   = gStartPsi   + MPC_N;
static size_t gStartCte   = gStartVel   + MPC_N;
static size_t gStartEpsi  = gStartCte   + MPC_N;
static size_t gStartDelta = gStartEpsi  + MPC_N;
static size_t gStartA     = gStartDelta + MPC_N - 1;

/* ========================= FG_EVAL IMPLEMENTATION ========================= */
class FG_eval
{
 public:
  /*! 
   * Fitted polynomial coefficients
   */
  Eigen::VectorXd vCoeffs;

  /* Constructor */
  FG_eval(Eigen::VectorXd vCoeffs) 
  { 
    this->vCoeffs = vCoeffs; 
  }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;

  /* `fg` is a vector containing the cost and constraints.
   `vars` is a vector containing the variable values (state & actuators) */
  void operator()(ADvector& fg, const ADvector& vars)
  {
    /* fg is a vector of constraints, x is a vector of constraints */
    fg[0] = 0.0;

    /* The part of the cost based on the reference state */
    for (int32_t i = 0; i < MPC_N; i++)
    {
      fg[0] += CppAD::pow(vars[gStartCte + i]  - MPC_REF_CTE,  2.0) * MPC_W_CTE;
      fg[0] += CppAD::pow(vars[gStartEpsi + i] - MPC_REF_EPSE, 2.0) * MPC_W_EPSI;
      fg[0] += CppAD::pow(vars[gStartVel + i]  - MPC_REF_VEL,  2.0) * MPC_W_VEL;
    }

    /* Minimize the use of actuators */
    for (int32_t i = 0; i < MPC_N - 1; i++)
    {
      fg[0] += CppAD::pow(vars[gStartDelta + i], 2.0) * MPC_W_STEER;
      fg[0] += CppAD::pow(vars[gStartA + i],     2.0) * MPC_W_THROT;
    }

    /* Minimize the value gap between sequential actuations */
    for (int32_t i = 0; i < MPC_N - 2; i++)
    {
      fg[0] += CppAD::pow(vars[gStartDelta + i + 1] - vars[gStartDelta + i], 2.0) * MPC_W_STEER_DIF;
      fg[0] += CppAD::pow(vars[gStartA + i + 1]     - vars[gStartA + i],     2.0) * MPC_W_THROT_DIF;
    }

    /** Setup Constraints **/
    /* NOTE: In this section you'll setup the model constraints.
     
       Initial constraints
      
       We add 1 to each of the starting indices due to cost being located at
       index 0 of `fg`.
       This bumps up the position of all the other values.
    */
    fg[1 + gStartX]    = vars[gStartX];
    fg[1 + gStartY]    = vars[gStartY];
    fg[1 + gStartPsi]  = vars[gStartPsi];
    fg[1 + gStartVel]  = vars[gStartVel];
    fg[1 + gStartCte]  = vars[gStartCte];
    fg[1 + gStartEpsi] = vars[gStartEpsi];

  /* The rest of the constraints */
    for (int32_t i = 0; i < MPC_N - 1; i++)
    {
      /* The state at time t+1 */
      const auto x1    = vars[gStartX + i + 1];
      const auto y1    = vars[gStartY + i + 1];
      const auto psi1  = vars[gStartPsi + i + 1];
      const auto v1    = vars[gStartVel + i + 1];
      const auto cte1  = vars[gStartCte + i + 1];
      const auto epsi1 = vars[gStartEpsi + i + 1];

      /* The state at time t */
      const auto x0    = vars[gStartX + i];
      const auto y0    = vars[gStartY + i];
      const auto psi0  = vars[gStartPsi + i];
      const auto v0    = vars[gStartVel + i];
      const auto cte0  = vars[gStartCte + i];
      const auto epsi0 = vars[gStartEpsi + i];

      // Only consider the actuation at time t.
      const auto delta0 = vars[gStartDelta + i];
      const auto a0     = vars[gStartA + i];

      const auto f0      = vCoeffs[0] + (vCoeffs[1] * x0) + (vCoeffs[2] * x0 * x0) + (vCoeffs[3] * x0 * x0 * x0);
      const auto psides0 = CppAD::atan(vCoeffs[1] + (2.0 * vCoeffs[2] * x0) +  (3.0 * vCoeffs[3] * x0 * x0));

      /* Here's `x` to get you started.
         The idea here is to constraint this value to be 0.
       
         Recall the equations for the model:
         x_[t+1]   = x[t] + v[t] * cos(psi[t]) * dt
         y_[t+1]   = y[t] + v[t] * sin(psi[t]) * dt
         psi_[t+1] = psi[t] + v[t] / Lf * delta[t] * dt
         v_[t+1]   = v[t] + a[t] * dt
         cte[t+1]  = f(x[t]) - y[t] + v[t] * sin(epsi[t]) * dt
         epsi[t+1] = psi[t] - psides[t] + v[t] * delta[t] / Lf * dt
      */
      const auto dLfFactor = (v0 / MPC_LF);
      fg[2 + gStartX + i]   = x1    - (x0 + v0 * CppAD::cos(psi0) * MPC_DT);
      fg[2 + gStartY + i]   = y1    - (y0 + v0 * CppAD::sin(psi0) * MPC_DT);
      fg[2 + gStartPsi + i] = psi1  - (psi0 + dLfFactor * delta0  * MPC_DT);
      fg[2 + gStartVel + i] = v1    - (v0 + a0 * MPC_DT);
      fg[2 + gStartCte + i] = cte1  - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * MPC_DT));
      fg[2 + gStartEpsi + i]= epsi1 - ((psi0 - psides0) + dLfFactor * delta0  * MPC_DT);
    }
  }
};

/* =========================== MPC IMPLEMENTATION ========================== */
MPC::MPC() {}
MPC::~MPC() {}

/*!
 * Solves the MPC
 */
vector<double> MPC::Solve(Eigen::VectorXd vState, Eigen::VectorXd vCoeffs)
{
  bool ok = true;
  size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  const double x    = vState[0];
  const double y    = vState[1];
  const double psi  = vState[2];
  const double v    = vState[3];
  const double cte  = vState[4];
  const double epsi = vState[5];

  const int32_t kDimState = 6;
  const int32_t kDimActuators = 2;
  const int32_t kNumVars = kDimState * MPC_N + kDimActuators * (MPC_N - 1);
  const int32_t kNumConstraints = kDimState * MPC_N;

  /* Initial value of the independent variables.
     SHOULD BE 0 besides initial state */
  Dvector vars(kNumVars);
  for (int32_t i = 0; i < kNumVars; i++)
  {
    vars[i] = 0;
  }

  /* Set the initial variable values */
  vars[gStartX]    = x;
  vars[gStartY]    = y;
  vars[gStartPsi]  = psi;
  vars[gStartVel]  = v;
  vars[gStartCte]  = cte;
  vars[gStartEpsi] = epsi;

  /* Lower and upper limits for x */
  Dvector vars_lowerbound(kNumVars);
  Dvector vars_upperbound(kNumVars);

  /* Lower and upper limits for the constraints
     Should be 0 besides initial state */
  for (int32_t i = 0; i < kNumConstraints; i++)
  {
    vars_lowerbound[i] = -1.0e19;
    vars_upperbound[i] =  1.0e19;
  }

  /* The upper and lower limits of delta are set to -25 and 25
     degrees (values in radians).
     NOTE: Feel free to change this to something else */
  for (int32_t i = gStartDelta; i < gStartA; i++)
  {
    vars_lowerbound[i] = -0.436332;
    vars_upperbound[i] =  0.436332;
  }

  /* Acceleration/decceleration upper and lower limits.
     NOTE: Feel free to change this to something else */
  for (int32_t i = gStartA; i < kNumVars; i++)
  {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] =  1.0;
  }

  /* Lower and upper limits for constraints
     All of these should be 0 except the initial
     state indices */
  Dvector constraints_lowerbound(kNumConstraints);
  Dvector constraints_upperbound(kNumConstraints);
  for (int32_t i = 0; i < kNumConstraints; i++)
  {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }
  constraints_lowerbound[gStartX]    = x;
  constraints_lowerbound[gStartY]    = y;
  constraints_lowerbound[gStartPsi]  = psi;
  constraints_lowerbound[gStartVel]  = v;
  constraints_lowerbound[gStartCte]  = cte;
  constraints_lowerbound[gStartEpsi] = epsi;

  constraints_upperbound[gStartX]    = x;
  constraints_upperbound[gStartY]    = y;
  constraints_upperbound[gStartPsi]  = psi;
  constraints_upperbound[gStartVel]  = v;
  constraints_upperbound[gStartCte]  = cte;
  constraints_upperbound[gStartEpsi] = epsi;

  /* object that computes objective and constraints */
  FG_eval fg_eval(vCoeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  this->vdPosX = {};
  this->vdPosY = {};

  for(int32_t i = 0; i < (MPC_N - 1); i++)
  {
    this->vdPosX.push_back(solution.x[gStartX+i]);
    this->vdPosY.push_back(solution.x[gStartY+i]);
  }

  // Return the first actuator values. The variables can be accessed with
  // `solution.x[i]`.
  //
  // {...} is shorthand for creating a vector, so auto x1 = {1.0,2.0}
  // creates a 2 element double vector.
  return {solution.x[gStartX + 1],   solution.x[gStartY + 1],
          solution.x[gStartPsi + 1], solution.x[gStartVel + 1],
          solution.x[gStartCte + 1], solution.x[gStartEpsi + 1],
          solution.x[gStartDelta],   solution.x[gStartA]};
}