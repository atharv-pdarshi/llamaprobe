#pragma once
#include <map>
#include <string>

// Minimal TOML subset parser: [sections] and key = value lines.
// String values may be quoted. CLI flags always override config.
class Config {
public:
    bool load(const std::string& path);  // returns false if file not found (not an error)

    std::string get_string(const std::string& section, const std::string& key,
                           const std::string& def = "") const;
    int         get_int   (const std::string& section, const std::string& key,
                           int def = 0) const;
    float       get_float (const std::string& section, const std::string& key,
                           float def = 0.f) const;
    bool        get_bool  (const std::string& section, const std::string& key,
                           bool def = false) const;

    bool has(const std::string& section, const std::string& key) const;

private:
    // data_[section][key] = raw string value
    std::map<std::string, std::map<std::string, std::string>> data_;
};
