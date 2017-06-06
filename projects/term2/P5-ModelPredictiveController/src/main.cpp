#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

/* Include the json namespace for convinience */
using json = nlohmann::json;

/* Helper functions to converting back and forth between radians and degrees */
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

/*!
 * Checks if the SocketIO event has JSON data.
 * If there is data the JSON object in string format will be returned,
 * else the empty string "" will be returned. */
string hasData(string s) 
{
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) 
  {
    return "";
  } 
  else if (b1 != string::npos && b2 != string::npos) 
  {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

/*!
 * Evaluate a polynomial
 */
double polyeval(Eigen::VectorXd coeffs, double x) 
{
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) 
  {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

/*! 
 * Fit a polynomial.
 * Adapted from
 * https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
 */
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order) 
{
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) 
  {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) 
  {
    for (int i = 0; i < order; i++) 
    {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

/*!
 * Application entry point
 */
int main() 
{
  uWS::Hub h;
  MPC mpc;

  /* On receiving a message */
  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) 
  {
    /* "42" at the start of the message means there's a websocket message event.
     * The 4 signifies a websocket message
     * The 2 signifies a websocket event */
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') 
    {
      string s = hasData(sdata);

      /* If in autonomous mode */
      if (s != "")
      {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry")
        {
          /* Read the telemetry data */
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];
          double str_ang = j[1]["steering_angle"];

          /* Convert to vehicle co-ordinate system */
          Eigen::VectorXd ptsx_vb;
          Eigen::VectorXd ptsy_vb;
          ptsx_vb = Eigen::VectorXd(ptsx.size());
          ptsy_vb = Eigen::VectorXd(ptsy.size());

          for (int i = 0; i < ptsx.size(); i++)
          {
            ptsx_vb[i] =  (ptsx[i] - px) * cos(psi) + (ptsy[i] - py) * sin(psi);
            ptsy_vb[i] = -(ptsx[i] - px) * sin(psi) + (ptsy[i] - py) * cos(psi);
          }

          /* Find the coefficients, by fitting a 3rd order polynomial */
          Eigen::VectorXd coeffs = polyfit(ptsx_vb, ptsy_vb, 3);

          /* Induce the required latency */
          double dt_lat = 0.100;
          double Lf     = 2.67;
          px  =  v*dt_lat;
          psi = -v*str_ang/Lf*dt_lat;

          /** Compute the error terms **/
          /* The cross track error is calculated by evaluating at polynomial at x, f(x)
            and subtracting y. x and y are at 0 to represent the car */
          double cte = polyeval(coeffs, px);

          /* Due to the sign starting at 0, the orientation error is -f'(x).
            derivative of coeffs[0] + coeffs[1] * x + coeffs[2] * x^2 + coeffs[3] * x^3 -> coeffs[1] 
            when x = 0
          */
          double epsi = -atan(coeffs[1]);

          /* Get the state in the car co-ordinate system */
          Eigen::VectorXd new_state(6);
          new_state << px, 0.0, psi, v, cte, epsi;

          /* Run solver to get the next state and the value of the actuators */
          double steer_value;
          double throttle_value;
          vector<double> next_state_act = mpc.Solve(new_state, coeffs);
          steer_value    = next_state_act[6];
          throttle_value = next_state_act[7];

          /* Output telemetry back */
          json msgJson;
          // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
          // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          msgJson["steering_angle"] =  -steer_value;
          msgJson["throttle"]       =  throttle_value;
          std::cout << "Steering Angle " << -steer_value   << std::endl;
          std::cout << "Throttle "       << throttle_value << std::endl;

          //Display the MPC predicted trajectory 
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;

          //Output from MPC solver
          mpc_x_vals  = mpc.mpc_x;
          mpc_y_vals =  mpc.mpc_y;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line
          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          //Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;
          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line
          for (int i = 0;  i < ptsx_vb.size();  i++)
          {   
            next_x_vals.push_back(ptsx_vb[i] ) ;
            next_y_vals.push_back(ptsy_vb[i] ) ;
          }
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      }
      /* If manual driving */
      else
      {
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  /* We don't need this since we're not using HTTP but if it's removed the
   program doesn't compile :-( */
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) 
  {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) 
    {
      res->end(s.data(), s.length());
    } 
    else 
    {
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) 
  {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length) 
  {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) 
  {
    std::cout << "Listening to port " << port << std::endl;
  } 
  else 
  {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}