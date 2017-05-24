#include "PID.h"
#include <iostream>

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() 
{
	
}

PID::~PID()
{

}

void PID::Init(double Kp, double Ki, double Kd, unsigned int int_len) 
{
	/* Set the flag */
	m_first_measurement = true;

	/* Save the values */
	m_Kp = Kp;
	m_Ki = Ki;
	m_Kd = Kd;

	/* Set the errors */
	m_tot_error = 0.0;

	/* Update the CTE values */
	m_cnt = 0;

	/* Resize the history */
	m_max_cte = int_len;
}

void PID::UpdateError(double cte) 
{
	/* If this is the first measurement */
	if (m_first_measurement == true)
	{
		/* Save the CTE value */
		m_prev_cte = cte;

		/* Reset the flag */
		m_first_measurement = false;
	}

	/* Append into the history list */
	m_cte_hist.push_back(cte);
	while (m_cte_hist.size() > m_max_cte)
	{
		m_cte_hist.erase(m_cte_hist.begin());
	}

	/* Compute the sum */
	double int_cte = 0.0;
	for(int i = 0; i < m_cte_hist.size(); i++)
	{
		int_cte += m_cte_hist[i];
	}

	/* Calculate the steering value */
	double diff_cte = cte - m_prev_cte;
	m_prev_cte = cte;
	m_steer = ((-m_Kp * cte) - (m_Kd * diff_cte) - (m_Ki * int_cte));

	/* Normalize the Steering */
	if(m_steer > 1.0)
	{
		m_steer = 1.0;
	}
	if(m_steer < -1.0)
	{
		m_steer = -1.0;
	}

	/* Compute the throttle */
	if (m_steer > 0.1)
	{
		m_throttle = -0.0;
	}
	else if (m_steer > 0.07)
	{
		m_throttle = 0.1;	
	}
	else if (m_steer > 0.03)
	{
		m_throttle = 0.3;	
	}
	else
	{
		m_throttle = 0.5;
	}

	/* Update the count */
	m_cnt++;

	/* Update the total error */
	m_tot_error += cte * cte;
}

double PID::getSteer(void)
{
	return m_steer;
}

double PID::getThrottle(void)
{
	return m_throttle;
}

double PID::TotalError() 
{
	return m_tot_error;
}

