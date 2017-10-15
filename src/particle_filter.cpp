/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;

	default_random_engine gen;
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];

	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	for (int i = 0; i < num_particles; ++i) {
		Particle *p = new Particle();
		p->id = i;
		p->x = dist_x(gen);
		p->y = dist_y(gen);
		p->theta = dist_theta(gen);
		p->weight = 1.0;
		particles.push_back(*p);

		weights.push_back(1.0);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
	std_x = std_pos[0];
	std_y = std_pos[1];
	std_theta = std_pos[2];


	for (int i = 0; i < particles.size(); ++i ) {
		Particle *p = &particles[i];
		double p_x = p->x;
		double p_y = p->y;
		double yaw = p->theta;

		normal_distribution<double> dist_x(0, std_x);
		normal_distribution<double> dist_y(0, std_y);
		normal_distribution<double> dist_theta(0, std_theta);

		if (fabs(yaw_rate) > 0.001) {
			p->x = p_x + velocity/yaw_rate * ( sin (yaw + yaw_rate*delta_t) - sin(yaw));
			p->y = p_y + velocity/yaw_rate * ( cos(yaw) - cos(yaw+yaw_rate*delta_t) );
			p->theta = yaw + yaw_rate*delta_t;
		}else {
			p->x = p_x + velocity*delta_t*cos(yaw);
			p->y = p_y + velocity*delta_t*sin(yaw);
		}
		
		p->x += dist_x(gen);
		p->y += dist_y(gen);
		p->theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i<observations.size(); ++i){
		double min_dist = -1;
		int min_j = -1;

		for (int j = 0; j<predicted.size(); ++j){
			double dist_ij = dist(observations[i].x, observations[i].y,
								predicted[j].x, predicted[j].y);
			if (dist_ij < min_dist || j == 0) {
				min_dist = dist_ij;
				min_j = j;
			}
		}
		observations[i].id = min_j;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];

	double gauss_norm = (1/(2 * M_PI * sig_x * sig_y));

	// transform
	for (int i = 0; i < particles.size(); ++i){
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		// transform each observation
		std::vector<LandmarkObs> observations_transformed;
		for (int o_id = 0; o_id < observations.size(); ++o_id){
			LandmarkObs new_m;

			double o_x = observations[o_id].x;
			double o_y = observations[o_id].y;

			new_m.x = p_x + cos(p_theta) * o_x - sin(p_theta) * o_y;
			new_m.y = p_y + sin(p_theta) * o_x + cos(p_theta) * o_y;
			observations_transformed.push_back(new_m);
		}

		// selected landmarks
		std::vector<LandmarkObs> selected_landmarks;
		for (int l = 0; l < map_landmarks.landmark_list.size(); ++l){
			LandmarkObs obs;
			obs.x = map_landmarks.landmark_list[l].x_f;
			obs.y = map_landmarks.landmark_list[l].y_f;
			if (dist(p_x, p_y, obs.x, obs.y) < sensor_range) {
				selected_landmarks.push_back(obs);
			}
		}

		dataAssociation(selected_landmarks, observations_transformed);

		double weight = 1;
		for (int o_id = 0; o_id < observations_transformed.size(); ++o_id){
			LandmarkObs *obs = &observations_transformed[o_id];
			double mu_x = selected_landmarks[observations_transformed[o_id].id].x;
			double mu_y = selected_landmarks[observations_transformed[o_id].id].y;
			double exponent = ( pow((obs->x - mu_x),2.0) )/(2 * pow(sig_x,2.0)) + ( pow((obs->y - mu_y),2.0) )/(2 * pow(sig_y,2.0));
			weight *= gauss_norm * exp(-exponent);
		}
		particles[i].weight = weight;
		weights[i] = weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<double> d(weights.begin(), weights.end());
	vector<Particle> new_particles;
	for (int i = 0; i < num_particles; ++i){
		new_particles.push_back(particles[d(gen)]);
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
