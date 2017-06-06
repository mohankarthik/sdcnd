#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

using namespace std;

#define MPC_LATENCY_S	(0.100d)
#define MPC_LATENCY_MS  (100)
#define MPC_LF			(2.67d)

#define MPC_SEC_TO_MS	(1000.0d)

class MPC 
{
 public:
  MPC();

  virtual ~MPC();

  vector<double> mpc_x;
  vector<double> mpc_y;

  // Solve the model given an initial state and polynomial coefficients.
  // Return the first actuatotions.
  vector<double> Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);
};

#endif /* MPC_H */