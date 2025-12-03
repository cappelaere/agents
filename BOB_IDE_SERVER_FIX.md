# Bob IDE Server Installation Fix

## Problem Summary

**Error**: `wget: unable to resolve host address 'bob-bot1.fyre.ibm.com'`

Bob IDE was unable to automatically install its custom VSCode Server on the remote VM (`rhel-pgc-3`) because:

1. The remote VM could not resolve the hostname `bob-bot1.fyre.ibm.com` via DNS
2. Even after adding the hostname to `/etc/hosts`, the remote VM could not reach the server due to firewall/network restrictions
3. The Bob IDE server at `bob-bot1.fyre.ibm.com:3000` was only accessible from the local machine, not from the remote VM

## Root Cause

The issue had **multiple layers**:

1. **DNS Resolution**: The remote VM's DNS servers could not resolve `bob-bot1.fyre.ibm.com`
2. **Network Connectivity**: Firewall rules blocked access from the remote VM to `bob-bot1.fyre.ibm.com:3000`
3. **Corrupted Installation**: Previous failed download left a 0-byte `vscode-server.tar.gz` file in `~/.bobide-server/bin/d15acc6639499cd7b7eae929475455048b05a4ba/`
4. **Wrong Directory**: Initial installation was in `.vscode-server` but Bob IDE actually uses `.bobide-server`

**Server Details**:
- Hostname: `bob-bot1.fyre.ibm.com`
- IP Address: `9.46.109.72`
- Port: `3000`
- Bob IDE Version: `1.105.1+bob0.0.11`
- Commit ID: `d15acc6639499cd7b7eae929475455048b05a4ba`

## Solution Implemented

### Step 1: Added hostname to /etc/hosts on remote VM
```bash
ssh rhel-pgc-3 "echo '9.46.109.72 bob-bot1.fyre.ibm.com' | sudo tee -a /etc/hosts"
```

### Step 2: Downloaded Bob IDE server on local machine
```bash
cd /tmp
curl -L -o bob-ide-server.tar.gz "http://bob-bot1.fyre.ibm.com:3000/reh/bob-ide/linux/x64/1.105.1+bob0.0.11"
```

The server redirects to IBM Cloud Object Storage:
```
https://bob-ide-executables.s3.us-south.cloud-object-storage.appdomain.cloud/bob-ide-reh-linux-x64-1.105.1%2Bbob0.0.11.tar.gz
```

### Step 3: Copied to remote VM
```bash
scp /tmp/bob-ide-server.tar.gz rhel-pgc-3:/tmp/bob-ide-server.tar.gz
```

### Step 4: Installed Bob IDE server
```bash
# First install to .vscode-server
ssh rhel-pgc-3 "
  mkdir -p ~/.vscode-server/bin/d15acc6639499cd7b7eae929475455048b05a4ba
  cd ~/.vscode-server/bin/d15acc6639499cd7b7eae929475455048b05a4ba
  tar -xzf /tmp/bob-ide-server.tar.gz
  touch .install-complete
"

# Then copy to .bobide-server (Bob IDE's actual location)
ssh rhel-pgc-3 "
  rm -rf ~/.bobide-server/bin/d15acc6639499cd7b7eae929475455048b05a4ba/*
  cp -r ~/.vscode-server/bin/d15acc6639499cd7b7eae929475455048b05a4ba/* \
        ~/.bobide-server/bin/d15acc6639499cd7b7eae929475455048b05a4ba/
  touch ~/.bobide-server/bin/d15acc6639499cd7b7eae929475455048b05a4ba/.install-complete
  rm /tmp/bob-ide-server.tar.gz
"
```

## Verification

The Bob IDE server is now installed at:
```
~/.bobide-server/bin/d15acc6639499cd7b7eae929475455048b05a4ba/
```

**Important**: Bob IDE uses `.bobide-server` directory, not `.vscode-server`!

Key files:
- `bin/bobide-server` - Main server executable
- `bin/remote-cli/bobide` - CLI tool
- `product.json` - Version and configuration
- `.install-complete` - Marker file for VSCode

## Critical Fix: Corrupted Installation Files

### Issue Discovered
After the initial installation, Bob IDE still failed with the same error. Investigation revealed:

```bash
$ ssh rhel-pgc-3 "ls -la ~/.bobide-server/bin/d15acc6639499cd7b7eae929475455048b05a4ba/"
-rw-r--r--. 1 vpcuser vpcuser 0 Nov 27 13:23 vscode-server.tar.gz
```

A **0-byte corrupted file** from a previous failed download was blocking the installation!

### Complete Fix Applied

```bash
# Remove corrupted files
ssh rhel-pgc-3 "rm -rf ~/.bobide-server/bin/d15acc6639499cd7b7eae929475455048b05a4ba/*"

# Copy working installation from .vscode-server to .bobide-server
ssh rhel-pgc-3 "
  cp -r ~/.vscode-server/bin/d15acc6639499cd7b7eae929475455048b05a4ba/* \
        ~/.bobide-server/bin/d15acc6639499cd7b7eae929475455048b05a4ba/
  touch ~/.bobide-server/bin/d15acc6639499cd7b7eae929475455048b05a4ba/.install-complete
"

# Verify installation
ssh rhel-pgc-3 "ls -la ~/.bobide-server/bin/d15acc6639499cd7b7eae929475455048b05a4ba/bin/bobide-server"
# Output: -rwxr-xr-x. 1 vpcuser vpcuser 911 Dec  3 15:46 .../bin/bobide-server ✅
```

### Key Lessons

1. **Bob IDE uses `.bobide-server` NOT `.vscode-server`**
2. **Failed downloads leave 0-byte files** that must be removed
3. **Always verify file sizes** after installation
4. **Check both directories** (.vscode-server and .bobide-server) when troubleshooting

## Testing

To test the connection:
1. Close any existing SSH connections to `rhel-pgc-3` in Bob IDE
2. Disconnect and reconnect to the remote host
3. Bob should now use the pre-installed server without attempting to download
4. Check the connection log - should show "start" without download errors

## Future Prevention

If this issue occurs again with a new Bob IDE version:

### Quick Diagnostic Checklist

1. **Check for corrupted files first**:
   ```bash
   ssh rhel-pgc-3 "find ~/.bobide-server/bin -name '*.tar.gz' -size 0"
   ```
   If found, remove them: `rm -f <path-to-0-byte-file>`

2. **Verify correct directory**:
   ```bash
   # Bob IDE uses .bobide-server, NOT .vscode-server
   ssh rhel-pgc-3 "ls -la ~/.bobide-server/bin/"
   ```

3. **Check if server is accessible from local machine**:
   ```bash
   curl -I http://bob-bot1.fyre.ibm.com:3000/reh/bob-ide/linux/x64/<VERSION>
   ```

### Complete Manual Installation Process

```bash
# 1. On local machine - Download Bob IDE server
curl -L -o /tmp/bob-ide-server.tar.gz "http://bob-bot1.fyre.ibm.com:3000/reh/bob-ide/linux/x64/<VERSION>"

# 2. Extract and get commit ID
cd /tmp
tar -xzf bob-ide-server.tar.gz ./product.json
COMMIT_ID=$(grep '"commit"' product.json | cut -d'"' -f4)
echo "Commit ID: $COMMIT_ID"

# 3. Copy to remote VM
scp /tmp/bob-ide-server.tar.gz rhel-pgc-3:/tmp/

# 4. Install on remote VM in BOTH directories (for safety)
ssh rhel-pgc-3 "
  # Install in .vscode-server
  mkdir -p ~/.vscode-server/bin/$COMMIT_ID
  cd ~/.vscode-server/bin/$COMMIT_ID
  tar -xzf /tmp/bob-ide-server.tar.gz
  touch .install-complete
  
  # Remove any corrupted files in .bobide-server
  rm -rf ~/.bobide-server/bin/$COMMIT_ID/*
  
  # Copy to .bobide-server (Bob IDE's actual location)
  mkdir -p ~/.bobide-server/bin/$COMMIT_ID
  cp -r ~/.vscode-server/bin/$COMMIT_ID/* ~/.bobide-server/bin/$COMMIT_ID/
  touch ~/.bobide-server/bin/$COMMIT_ID/.install-complete
  
  # Cleanup
  rm /tmp/bob-ide-server.tar.gz
"

# 5. Verify installation
ssh rhel-pgc-3 "ls -lh ~/.bobide-server/bin/$COMMIT_ID/bin/bobide-server"
```

### Troubleshooting Tips

- **Always check file sizes**: 0-byte files indicate failed downloads
- **Check both directories**: `.vscode-server` AND `.bobide-server`
- **Remove corrupted files**: Don't try to overwrite, delete first
- **Verify executables**: Ensure `bin/bobide-server` exists and is executable

## Network Configuration Details

### Remote VM DNS Configuration
```
nameserver 161.26.0.10
nameserver 161.26.0.11
nameserver 9.0.128.50
nameserver 9.0.130.50
```

### /etc/hosts Entry Added
```
9.46.109.72 bob-bot1.fyre.ibm.com
```

## Alternative Solutions Considered

1. **Configure VPN/Network Access**: Would require network team involvement
2. **Use Direct Cloud Storage URL**: Not feasible as URLs are signed and expire
3. **Mirror Bob IDE Server**: Would require maintaining a separate server
4. **Manual Installation** (Chosen): Most practical immediate solution

## Date
December 3, 2025

## Status
✅ **RESOLVED** - Bob IDE server successfully installed on remote VM