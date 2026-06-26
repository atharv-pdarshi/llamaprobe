#include "config.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cctype>

// Trim whitespace from both ends of a string
static std::string trim(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])))
        ++start;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1])))
        --end;
    return s.substr(start, end - start);
}

// Strip surrounding quotes from a value string (single or double)
static std::string strip_quotes(const std::string& s) {
    if (s.size() >= 2 &&
        ((s.front() == '"' && s.back() == '"') ||
         (s.front() == '\'' && s.back() == '\''))) {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

bool Config::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        return false;  // not found — not a fatal error
    }

    std::string current_section;
    std::string line;
    while (std::getline(f, line)) {
        // Strip inline comments (# not inside quotes — simple approximation)
        size_t hash_pos = line.find('#');
        if (hash_pos != std::string::npos) {
            line = line.substr(0, hash_pos);
        }

        std::string t = trim(line);
        if (t.empty()) continue;

        // Section header: [section_name]
        if (t.front() == '[' && t.back() == ']') {
            current_section = trim(t.substr(1, t.size() - 2));
            continue;
        }

        // Key = value
        size_t eq = t.find('=');
        if (eq == std::string::npos) continue;

        std::string key   = trim(t.substr(0, eq));
        std::string value = trim(t.substr(eq + 1));
        value = strip_quotes(value);

        if (!key.empty()) {
            data_[current_section][key] = value;
        }
    }

    return true;
}

bool Config::has(const std::string& section, const std::string& key) const {
    auto sit = data_.find(section);
    if (sit == data_.end()) return false;
    return sit->second.find(key) != sit->second.end();
}

std::string Config::get_string(const std::string& section, const std::string& key,
                                const std::string& def) const {
    auto sit = data_.find(section);
    if (sit == data_.end()) return def;
    auto kit = sit->second.find(key);
    if (kit == sit->second.end()) return def;
    return kit->second;
}

int Config::get_int(const std::string& section, const std::string& key, int def) const {
    auto sit = data_.find(section);
    if (sit == data_.end()) return def;
    auto kit = sit->second.find(key);
    if (kit == sit->second.end()) return def;
    try {
        return std::stoi(kit->second);
    } catch (...) {
        return def;
    }
}

float Config::get_float(const std::string& section, const std::string& key, float def) const {
    auto sit = data_.find(section);
    if (sit == data_.end()) return def;
    auto kit = sit->second.find(key);
    if (kit == sit->second.end()) return def;
    try {
        return std::stof(kit->second);
    } catch (...) {
        return def;
    }
}

bool Config::get_bool(const std::string& section, const std::string& key, bool def) const {
    auto sit = data_.find(section);
    if (sit == data_.end()) return def;
    auto kit = sit->second.find(key);
    if (kit == sit->second.end()) return def;
    const std::string& v = kit->second;
    if (v == "true" || v == "1" || v == "yes") return true;
    if (v == "false" || v == "0" || v == "no") return false;
    return def;
}
