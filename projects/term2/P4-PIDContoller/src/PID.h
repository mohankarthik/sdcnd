#ifndef PID_H
#define PID_H

#include <vector>
using namespace std;

class PID 
{
private:
  bool m_first_measurement; /*!< Tracks if this is the first measurement or not */

  double m_prev_cte; /*!< Previous CTE values */
  vector <double> m_cte_hist; /*!< Maintains the history of all CTEs */
  unsigned int m_max_cte; /*!< Maximum number of CTEs to maintain in history */
  unsigned long long m_cnt; /*!< Number of measurements */

  double m_tot_error; /*!< Keeps track of the total error so far */

  /*
   * Coefficients
   */ 
  double m_Kp;
  double m_Ki;
  double m_Kd;

  double m_steer; /*!< Current steering prediction */
  double m_throttle;

public:
  /*
   * Constructor
   */
  PID();

  /*
   * Destructor.
   */
  virtual ~PID();

  /*
   * Initialize PID.
   */
  void Init(double Kp, double Ki, double Kd, unsigned int int_len);

  /*
   * Update the PID error variables given cross track error.
   */
  void UpdateError(double cte);

  /*!
   * Computes the steering angle
   */
  double getSteer(void);
  double getThrottle(void);

  /*
  * Calculate the total PID error.
  */
  double TotalError();
};

#endif /* PID_H */
