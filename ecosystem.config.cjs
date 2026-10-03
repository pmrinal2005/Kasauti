module.exports = {
  apps: [{ name: "kasauti", script: "node_modules/next/dist/bin/next", args: "start -p 3000 -H 0.0.0.0", env: { NODE_ENV: "production" }, watch: false, instances: 1, exec_mode: "fork" }],
};
