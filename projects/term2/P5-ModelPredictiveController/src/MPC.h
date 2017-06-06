#ifndef MPC_H
#define MPC_H

/* =============================== INCLUDES ================================ */
#include <vector>
#include "Eigen-3.3/Eigen/Core"

/* ============================== NAMESPACES =============================== */
using namespace std;

/* ================================ DEFINES ================================ */
#define MPC_LATENCY_S	(0.100d)/*!< The latency to be induced in seconds */
#define MPC_LATENCY_MS  (100) 	/*!< The latency to be induced in milliseconds */
#define MPC_LF			(2.67d)	/*!< The LF factor */

#define MPC_N 			(15)	/*!< The number of time steps to predict in the future */
#define MPC_DT			(0.10)	/*!< The value of each time step */

#define MPC_REF_CTE		(0.0)	/*!< The ideal CTE value */
#define MPC_REF_EPSE	(0.0)	/*!< The ideal EPSI value */
#define MPC_REF_VEL		(40.0)	/*!< The ideal velocity */

#define MPC_W_CTE		(1.0)	/*!< The weight of the CTE cost */
#define MPC_W_EPSI		(1.0)	/*!< The weight of the EPSE cost */
#define MPC_W_VEL		(1.0)	/*!< The weight of the velocity cost */
#define MPC_W_STEER		(125.0)	/*!< The weight of the steering actuator */
#define MPC_W_THROT		(1.0)	/*!< The weight of the throttle actuator */ 
#define MPC_W_STEER_DIF	(1.0)	/*!< The weight of the steering difference */
#define MPC_W_THROT_DIF	(1.0)	/*!< The weight of the throttle difference */

/* ============================ CLASS DEFINITION =========================== */
class MPC 
{
 public:
  MPC();

  virtual ~MPC();

  vector<double> vdPosX;
  vector<double> vdPosY;

  // Solve the model given an initial state and polynomial coefficients.
  // Return the first actuatotions.
  vector<double> Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);
};

#endif /* MPC_H */