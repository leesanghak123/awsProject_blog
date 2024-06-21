package com.cos.blog.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import com.cos.blog.model.Board;
import com.cos.blog.model.Gpt;

public interface GptRepository extends JpaRepository<Gpt, Integer>{
	Page<Gpt> findByUserUsername(String username, Pageable pageable);
}