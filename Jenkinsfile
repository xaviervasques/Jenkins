pipeline {
  agent {
    dockerfile {
      filename 'Dockerfile'
    }

  }
  stages {
    stage('Initialize') {
      steps {
        sh 'echo "Starting build"'
      }
    }

    stage('Build') {
      steps {
        sh 'build docker -t mlops-test .'
      }
    }

  }
}