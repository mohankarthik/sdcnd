#ifndef MPC_H
#define MPC_H

/* ==================== INCLUDES ==================== */
#include <vector>
#include "Eigen-3.3/Eigen/Core"

/* =================== NAMESPACES =================== */
using namespace std;

/* ================ CLASS DEFINITION ================ */
class MPC 
{
 public:
  MPC();

  virtual ~MPC();

  /*! 
   * Solve the model given an initial state and polynomial coefficients
   * Return the first actuatotions
   */
  vector<double> Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);

  /*!
   * Gets the latest X and Y tragectories
   */
  vector<double> getXs();
  vector<double> getYs();
  
 private:
  vector<double> gvSolX;
};

#endif /* MPC_H */
